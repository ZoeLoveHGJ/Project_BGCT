import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


from Framework import (
    TraditionalAlgorithmInterface,
    AlgorithmStepResult,
    Tag,
    CONSTANTS
)


@dataclass
class LAPCTState:
    """
    LAPCT 任务状态。
    存储当前分支下的标签集合以及搜索深度。
    """
    tags_to_identify: List[Tag] = field(default_factory=list)
    depth: int = 0


class LAPCTAlgorithm(TraditionalAlgorithmInterface):

    def __init__(self, tags_in_field: List[Tag], **kwargs):

        super().__init__(tags_in_field, **kwargs)

        initial_state = LAPCTState(
            tags_to_identify=self.tags_in_field, depth=0)
        self.query_stack: List[LAPCTState] = [initial_state]

        self.total_tag_count = len(tags_in_field)
        self.k_threshold_divisor = kwargs.get('k_threshold_divisor', 3.0)

        self.metrics['total_tag_responses'] = 0

        self.id_length = len(tags_in_field[0].id) if tags_in_field else 96
        self.tag_response_counts: Dict[str, int] = {
            t.id: 0 for t in tags_in_field}

        self.enable_monitoring = kwargs.get(
            'enable_resource_monitoring', False)

    def _create_step_result(self, *args, **kwargs) -> AlgorithmStepResult:
        """辅助函数：创建结果对象并附加资源监控指标。"""
        result = AlgorithmStepResult(*args, **kwargs)
        if self.enable_monitoring:
            result.internal_metrics = {'stack_depth': len(self.query_stack)}
        return result

    def is_finished(self) -> bool:
        """判断是否完成识别。"""
        finished = len(self.identified_tags) == self.total_tag_count

        if not self.query_stack and not finished:
            remaining_tags = [
                t for t in self.tags_in_field if t.id not in self.identified_tags]
            if remaining_tags:
                self.query_stack.append(LAPCTState(
                    tags_to_identify=remaining_tags, depth=0))
                return False

        if finished and 'avg_tag_responses' not in self.metrics:
            if self.total_tag_count > 0:
                self.metrics['avg_tag_responses'] = self.metrics['total_tag_responses'] / \
                    self.total_tag_count
            else:
                self.metrics['avg_tag_responses'] = 0
        return finished

    def perform_step(self) -> AlgorithmStepResult:
        """
        执行一步 LAPCT 协议。
        包含：发送查询 -> 确定分裂策略 -> 并行接收多个子时隙响应。
        """
        if not self.query_stack:
            return self._create_step_result(operation_type='internal_op')

        current_state = self.query_stack.pop(0)

        candidates = [
            t for t in current_state.tags_to_identify if t.id not in self.identified_tags]
        active_tags_query = self.channel.filter_active_tags(candidates)

        if not active_tags_query:
            self.metrics['idle_slots'] += 1
            return self._create_step_result('idle_slot', CONSTANTS.READER_CMD_BASE_BITS)

        observed_signal, _ = self.channel.resolve_collision(active_tags_query)

        collision_indices = [i for i, bit in enumerate(
            observed_signal) if bit == 'X']

        reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(observed_signal)
        total_tag_bits = 0
        total_time_overhead_us = 0.0

        t_query = reader_bits / CONSTANTS.READER_BITS_PER_US + CONSTANTS.T1_US
        total_time_overhead_us += t_query

        if not collision_indices:

            matched_tag = next(
                (t for t in active_tags_query if t.id == observed_signal), None)

            if matched_tag:

                self.metrics['total_tag_responses'] += 1
                self.tag_response_counts[matched_tag.id] += 1

                self.identified_tags.add(matched_tag.id)
                self.metrics['success_slots'] += 1

                resp_bits = self.id_length
                total_tag_bits += resp_bits
                total_time_overhead_us += (resp_bits /
                                           CONSTANTS.TAG_BITS_PER_US + CONSTANTS.T2_MIN_US)

                return self._create_step_result(
                    'success_slot', reader_bits, total_tag_bits, self.id_length,
                    override_time_us=total_time_overhead_us
                )
            else:

                self.metrics['total_tag_responses'] += len(active_tags_query)
                for t in active_tags_query:
                    self.tag_response_counts[t.id] += 1

                self.metrics['collision_slots'] += 1
                self.query_stack.insert(0, LAPCTState(
                    active_tags_query, current_state.depth))

                total_time_overhead_us += (self.id_length /
                                           CONSTANTS.TAG_BITS_PER_US + CONSTANTS.T2_MIN_US)
                return self._create_step_result(
                    'collision_slot', reader_bits, self.id_length, self.id_length,
                    override_time_us=total_time_overhead_us,
                    operation_description="CRC Fail/Retry"
                )

        num_tags_est = len(active_tags_query)
        k_threshold = 0
        if num_tags_est >= self.k_threshold_divisor:
            k_threshold = math.floor(
                math.log(num_tags_est / self.k_threshold_divisor, 4))

        use_4_way = (current_state.depth <=
                     k_threshold and len(collision_indices) >= 2)

        sub_groups_map = {}
        c1 = collision_indices[0]

        if use_4_way:
            c2 = collision_indices[1]
            target_features = ['00', '01', '10', '11']
            for tag in active_tags_query:
                key = tag.id[c1] + tag.id[c2]
                if key not in sub_groups_map:
                    sub_groups_map[key] = []
                sub_groups_map[key].append(tag)
        else:
            target_features = ['0', '1']
            for tag in active_tags_query:
                key = tag.id[c1]
                if key not in sub_groups_map:
                    sub_groups_map[key] = []
                sub_groups_map[key].append(tag)

        sub_slot_count = len(target_features)

        for feature in target_features:
            sub_candidates = sub_groups_map.get(feature, [])

            if sub_candidates:
                self.metrics['total_tag_responses'] += len(sub_candidates)
                for t in sub_candidates:
                    self.tag_response_counts[t.id] += 1

            sub_obs_signal, _ = self.channel.resolve_collision(sub_candidates)

            slot_bits = self.id_length
            total_tag_bits += (len(sub_candidates) * slot_bits)
            total_time_overhead_us += (slot_bits /
                                       CONSTANTS.TAG_BITS_PER_US + CONSTANTS.T2_MIN_US)

            if not sub_candidates:
                self.metrics['idle_slots'] += 1
            elif 'X' in sub_obs_signal:
                self.metrics['collision_slots'] += 1
                self.query_stack.insert(0, LAPCTState(
                    sub_candidates, current_state.depth + 1))
            else:

                matched = next(
                    (t for t in sub_candidates if t.id == sub_obs_signal), None)
                if matched:
                    self.identified_tags.add(matched.id)
                    self.metrics['success_slots'] += 1
                    residues = [
                        t for t in sub_candidates if t.id != matched.id]
                    if residues:
                        self.query_stack.insert(0, LAPCTState(
                            residues, current_state.depth + 1))
                else:
                    self.metrics['collision_slots'] += 1
                    self.query_stack.insert(0, LAPCTState(
                        sub_candidates, current_state.depth + 1))

        return self._create_step_result(
            'collision_slot',
            reader_bits,
            total_tag_bits,
            self.id_length * sub_slot_count,
            override_time_us=total_time_overhead_us,
            operation_description=f"Parallel Split ({'4-way' if use_4_way else '2-way'})"
        )
