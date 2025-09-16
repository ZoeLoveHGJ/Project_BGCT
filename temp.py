import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set


from Framework import (
    TraditionalAlgorithmInterface,
    AlgorithmStepResult,
    Tag,
    CONSTANTS,
    apply_ber_noise
)
from Tool import RfidUtils


@dataclass
class BGVCTState:
    tags_to_identify: List[Tag] = field(default_factory=list)


class BGVCT_Algorithm(TraditionalAlgorithmInterface):

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field, **kwargs)
        self.d_max = kwargs.get('d_max', 8)
        initial_state = BGVCTState(tags_to_identify=self.tags_in_field)
        self.query_stack: List[BGVCTState] = [initial_state]
        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0
        self.tag_response_counts: Dict[str, int] = {
            t.id: 0 for t in tags_in_field}
        self.current_mode = 'PLANNING_PHASE'
        self.tags_for_planning: List[Tag] = []
        self.planned_groups: List[List[Tag]] = []
        self.non_idle_groups: List[List[Tag]] = []
        self.sub_slot_cursor: int = 0
        self.current_split_prefix_len: int = 0
        self.current_d_to_use: int = 0
        self.pending_cmd_reader_bits: float = 0.0

        self.enable_monitoring = kwargs.get(
            'enable_resource_monitoring', False)

    def _create_step_result(self, *args, **kwargs) -> AlgorithmStepResult:
        result = AlgorithmStepResult(*args, **kwargs)
        if self.enable_monitoring:
            result.internal_metrics = {'stack_depth': len(self.query_stack)}
        return result

    def is_finished(self) -> bool:
        finished = len(self.identified_tags) == len(self.tags_in_field)
        if finished and 'avg_tag_responses' not in self.metrics:
            counts = list(self.tag_response_counts.values())
            self.metrics['avg_tag_responses'] = np.mean(
                counts) if counts else 0
        return finished

    def _get_collision_info_cost(self, tags_to_process: List[Tag]) -> AlgorithmStepResult:
        self.metrics['collision_slots'] += 1
        num_tags = len(tags_to_process)
        for tag in tags_to_process:
            self.tag_response_counts[tag.id] += 1
        tag_ids = [t.id for t in tags_to_process]
        common_prefix, _ = RfidUtils.get_collision_info(tag_ids)
        prefix_len = len(common_prefix)
        remaining_len = self.id_length - prefix_len
        reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS + prefix_len
        tag_bits_cost = num_tags * remaining_len
        return self._create_step_result('collision_slot', reader_bits=reader_bits_cost, tag_bits=tag_bits_cost,
                                        expected_max_tag_bits=remaining_len,
                                        operation_description=f"信息获取碰撞: {num_tags} 个标签")

    def _handle_final_identification(self, tag: Tag, prefix_len: int = 0) -> AlgorithmStepResult:
        """
        处理一个单标签的最终识别时隙，包含BER判断。
        """
        self.tag_response_counts[tag.id] += 1
        remaining_len = self.id_length - prefix_len

        perfect_response = tag.id[prefix_len:]
        noisy_response = apply_ber_noise(perfect_response, self.ber)

        reader_bits = CONSTANTS.READER_CMD_BASE_BITS + prefix_len

        if perfect_response == noisy_response:
            self.identified_tags.add(tag.id)
            self.metrics['success_slots'] += 1
            return self._create_step_result('success_slot', reader_bits, remaining_len, remaining_len)
        else:
            self.query_stack.insert(0, BGVCTState(tags_to_identify=[tag]))
            self.metrics['collision_slots'] += 1
            return self._create_step_result('collision_slot', reader_bits, remaining_len, remaining_len,
                                            operation_description="Success slot failed due to BER")

    def perform_step(self) -> AlgorithmStepResult:
        if self.current_mode == 'PLANNING_PHASE':
            if not self.query_stack:
                if not self.is_finished():
                    self.is_finished()
                return self._create_step_result('internal_op', operation_description="完成")

            current_state = self.query_stack.pop(0)
            tags = [
                t for t in current_state.tags_to_identify if t.id not in self.identified_tags]

            if not tags:
                return self._create_step_result('internal_op', operation_description="空状态弹出")

            if len(tags) == 1:

                return self._handle_final_identification(tags[0], prefix_len=0)

            self.tags_for_planning = tags
            self.current_mode = 'AWAITING_PLANNING_DATA'
            return self._get_collision_info_cost(self.tags_for_planning)

        elif self.current_mode == 'AWAITING_PLANNING_DATA':
            tags = self.tags_for_planning
            tag_ids = [t.id for t in tags]
            common_prefix, collision_positions = RfidUtils.get_collision_info(
                tag_ids)
            d_detected = len(collision_positions)
            if d_detected == 0:
                if tags:
                    for tag in tags:
                        self.identified_tags.add(tag.id)
                self.current_mode = 'PLANNING_PHASE'
                return self._create_step_result('internal_op', operation_description="处理重复ID")

            d_to_use = min(d_detected, self.d_max)
            indices = np.linspace(0, d_detected - 1, d_to_use, dtype=int)
            positions_to_use = [collision_positions[i] for i in indices]
            groups = [[] for _ in range(2**d_to_use)]
            for tag in tags:
                feature = "".join([tag.id[i] for i in positions_to_use])
                groups[int(feature, 2)].append(tag)

            self.planned_groups = groups
            self.current_d_to_use = d_to_use
            self.current_split_prefix_len = len(common_prefix)
            self.pending_cmd_reader_bits = CONSTANTS.READER_CMD_BASE_BITS + \
                len(common_prefix) + 5 + d_to_use * 7
            self.current_mode = 'PREDICTION_PHASE'
            return self._create_step_result('internal_op', operation_description=f"规划完成, d={d_to_use}")

        elif self.current_mode == 'PREDICTION_PHASE':
            self.non_idle_groups = [g for g in self.planned_groups if g]
            if not self.non_idle_groups:
                self.current_mode = 'PLANNING_PHASE'
                return self._create_step_result('internal_op', operation_description="预测结果：所有时隙均为空")

            self.current_mode = 'EXECUTING_PHASE'
            self.sub_slot_cursor = 0
            bitmap_len = 2**self.current_d_to_use
            reader_bits = self.pending_cmd_reader_bits
            self.pending_cmd_reader_bits = 0.0

            return self._create_step_result('collision_slot', reader_bits=reader_bits, tag_bits=bitmap_len,
                                            expected_max_tag_bits=bitmap_len, operation_description=f"接收预测位图 (长度={bitmap_len})")

        elif self.current_mode == 'EXECUTING_PHASE':
            group = self.non_idle_groups[self.sub_slot_cursor]
            self.sub_slot_cursor += 1

            result = None

            if len(group) == 1:
                result = self._handle_final_identification(
                    group[0], self.current_split_prefix_len)
            else:
                self.query_stack.insert(0, BGVCTState(tags_to_identify=group))
                self.metrics['collision_slots'] += 1
                expected_bits = self.id_length - self.current_split_prefix_len
                reader_bits_cost = CONSTANTS.QUERYREP_CMD_BITS
                result = self._create_step_result('collision_slot', reader_bits=reader_bits_cost,
                                                  tag_bits=len(
                                                      group) * expected_bits,
                                                  expected_max_tag_bits=expected_bits)

            if self.sub_slot_cursor >= len(self.non_idle_groups):
                self.current_mode = 'PLANNING_PHASE'

            return result

        return self._create_step_result('internal_op', operation_description="错误: 未知状态")


class BGCT(BGVCT_Algorithm):

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field, **kwargs)
        self.chunk_size = kwargs.get('chunk_size', 16)
        self.d_target = kwargs.get('d_target', 8)
        self.probe_offset = 0
        self.accumulated_positions = []

    def perform_step(self) -> AlgorithmStepResult:

        if self.current_mode not in ['PLANNING_PHASE', 'PROGRESSIVE_PROBE']:
            return super().perform_step()

        if self.current_mode == 'PLANNING_PHASE':
            if not self.query_stack:
                if not self.is_finished():
                    self.is_finished()
                return self._create_step_result('internal_op', operation_description="完成")

            current_state = self.query_stack.pop(0)
            tags = [
                t for t in current_state.tags_to_identify if t.id not in self.identified_tags]

            if not tags:
                return self._create_step_result('internal_op')

            if len(tags) == 1:
                return self._handle_final_identification(tags[0], prefix_len=0)

            self.tags_for_planning = tags
            tag_ids = [t.id for t in self.tags_for_planning]
            common_prefix, _ = RfidUtils.get_collision_info(tag_ids)
            self.probe_offset = len(common_prefix)
            self.accumulated_positions = []
            self.current_mode = 'PROGRESSIVE_PROBE'
            return self._create_step_result('internal_op', operation_description="启动渐进式探测")

        elif self.current_mode == 'PROGRESSIVE_PROBE':
            if self.probe_offset >= self.id_length or len(self.accumulated_positions) >= self.d_target:
                return self._process_accumulated_info()

            tag_ids = [t.id for t in self.tags_for_planning]
            start, end = self.probe_offset, min(
                self.probe_offset + self.chunk_size, self.id_length)
            current_chunk_len = end - start

            if current_chunk_len <= 0:
                return self._process_accumulated_info()

            perfect_chunks = [tid[start:end] for tid in tag_ids]

            noisy_chunks = [apply_ber_noise(chunk, self.ber)
                            for chunk in perfect_chunks]
            _, chunk_collision_indices = RfidUtils.get_collision_info(
                noisy_chunks)

            for rel_idx in chunk_collision_indices:
                abs_idx = start + rel_idx
                if abs_idx not in self.accumulated_positions:
                    self.accumulated_positions.append(abs_idx)

            self.probe_offset = end

            self.metrics['collision_slots'] += 1
            for tag in self.tags_for_planning:
                self.tag_response_counts[tag.id] += 1

            common_prefix_len = start
            reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS + common_prefix_len
            tag_bits_cost = len(self.tags_for_planning) * current_chunk_len

            return self._create_step_result('collision_slot', reader_bits=reader_bits_cost, tag_bits=tag_bits_cost,
                                            expected_max_tag_bits=current_chunk_len,
                                            operation_description=f"块探测 (pos:{start}-{end})")

        return self._create_step_result('internal_op', operation_description="错误: 未知状态")

    def _process_accumulated_info(self) -> AlgorithmStepResult:
        tags = self.tags_for_planning
        tag_ids = [t.id for t in tags]
        common_prefix, _ = RfidUtils.get_collision_info(tag_ids)

        collision_positions = self.accumulated_positions
        d_detected = len(collision_positions)

        if d_detected == 0:
            if tags:
                for tag in tags:
                    self.identified_tags.add(tag.id)
            self.current_mode = 'PLANNING_PHASE'
            return self._create_step_result('internal_op', operation_description="处理重复ID")

        d_to_use = min(d_detected, self.d_target)
        positions_to_use = collision_positions[:d_to_use]

        groups = [[] for _ in range(2**d_to_use)]
        for tag in tags:
            feature = "".join([tag.id[i] for i in positions_to_use])
            groups[int(feature, 2)].append(tag)

        self.planned_groups = groups
        self.current_d_to_use = d_to_use
        self.current_split_prefix_len = len(common_prefix)
        self.pending_cmd_reader_bits = CONSTANTS.READER_CMD_BASE_BITS + \
            len(common_prefix) + 5 + d_to_use * 7
        self.current_mode = 'PREDICTION_PHASE'

        return self._create_step_result('internal_op', operation_description=f"规划完成 (渐进式 d={d_to_use})")
