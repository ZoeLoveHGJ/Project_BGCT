
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


class RfidUtils:
    @staticmethod
    def get_collision_info(tag_ids: List[str]) -> (str, List[int]):
        if not tag_ids:
            return "", []
        min_len = min(len(tid) for tid in tag_ids)
        common_prefix = ""
        for i in range(min_len):
            char = tag_ids[0][i]
            if all(tid[i] == char for tid in tag_ids):
                common_prefix += char
            else:
                break
        return common_prefix, []


@dataclass
class DQTAState:
    """用于在主堆栈中存储待处理的标签集合。"""
    tags_to_identify: List[Tag] = field(default_factory=list)


class DQTAAlgorithm(TraditionalAlgorithmInterface):
    """DQTA 算法的复现实现 - V2 适配版"""

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)
        self.k_max = kwargs.get('k_max', 3)
        initial_state = DQTAState(tags_to_identify=self.tags_in_field)
        self.query_stack: List[DQTAState] = [initial_state]
        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0
        self.tag_response_counts: Dict[str, int] = {
            t.id: 0 for t in tags_in_field}
        self.current_mode = 'PLANNING_PHASE'
        self.tags_for_planning: List[Tag] = []
        self.planned_groups: List[List[Tag]] = []
        self.sub_slot_cursor: int = 0
        self.current_split_prefix_len: int = 0
        self.current_k_to_use: int = 0
        self.pending_cmd_reader_bits: float = 0.0

        self.ber = kwargs.get('ber', 0.0)
        self.enable_monitoring = kwargs.get(
            'enable_resource_monitoring', False)

    def _create_step_result(self, *args, **kwargs) -> AlgorithmStepResult:
        """
        V2 新增: 辅助函数，用于统一创建 AlgorithmStepResult 对象。
        如果资源监控开启，此函数会自动附加内部状态指标。
        """
        result = AlgorithmStepResult(*args, **kwargs)
        if self.enable_monitoring:
            result.internal_metrics = {'stack_depth': len(self.query_stack)}
        return result

    def _get_consecutive_collision_info(self, tag_ids: List[str], prefix_len: int) -> int:
        """DQTA的核心辅助函数：检测第一个碰撞位之后连续的碰撞位数。"""
        if len(tag_ids) <= 1:
            return 0
        id_len = len(tag_ids[0])
        first_coll_pos = -1
        for i in range(prefix_len, id_len):
            if len(set(tag[i] for tag in tag_ids)) > 1:
                first_coll_pos = i
                break
        if first_coll_pos == -1:
            return 0
        k_detected = 1
        for i in range(first_coll_pos + 1, id_len):
            if len(set(tag[i] for tag in tag_ids)) > 1:
                k_detected += 1
            else:
                break
        return k_detected

    def is_finished(self) -> bool:
        finished = len(self.identified_tags) == len(self.tags_in_field)
        if finished and 'avg_tag_responses' not in self.metrics:
            counts = list(self.tag_response_counts.values())
            self.metrics['avg_tag_responses'] = np.mean(
                counts) if counts else 0
        return finished

    def perform_step(self) -> AlgorithmStepResult:
        if self.current_mode == 'PLANNING_PHASE':
            if not self.query_stack:
                return self._create_step_result('internal_op', operation_description="Finished")

            current_state = self.query_stack.pop(0)
            tags = [
                t for t in current_state.tags_to_identify if t.id not in self.identified_tags]

            if not tags:
                return self._create_step_result('internal_op', operation_description="Empty state popped")

            common_prefix, _ = RfidUtils.get_collision_info(
                [t.id for t in tags])

            if len(tags) == 1:
                tag = tags[0]
                self.identified_tags.add(tag.id)
                self.metrics['success_slots'] += 1
                self.tag_response_counts[tag.id] += 1
                reader_bits = CONSTANTS.READER_CMD_BASE_BITS + \
                    len(common_prefix)
                tag_bits = self.id_length - len(common_prefix)
                return self._create_step_result('success_slot', reader_bits, tag_bits, tag_bits)

            self.tags_for_planning = tags
            self.current_mode = 'AWAITING_PLANNING_DATA'

            self.metrics['collision_slots'] += 1
            for tag in self.tags_for_planning:
                self.tag_response_counts[tag.id] += 1

            prefix_len = len(common_prefix)
            remaining_len = self.id_length - prefix_len
            reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS + prefix_len
            tag_bits_cost = len(tags) * remaining_len

            return self._create_step_result('collision_slot',
                                            reader_bits=reader_bits_cost,
                                            tag_bits=tag_bits_cost,
                                            expected_max_tag_bits=remaining_len,
                                            operation_description=f"Initial collision for DQTA info")

        elif self.current_mode == 'AWAITING_PLANNING_DATA':
            tags = self.tags_for_planning
            tag_ids = [t.id for t in tags]
            common_prefix, _ = RfidUtils.get_collision_info(tag_ids)
            prefix_len = len(common_prefix)

            k_detected = self._get_consecutive_collision_info(
                tag_ids, prefix_len)
            k_to_use = min(k_detected, self.k_max)
            if k_to_use == 0:
                k_to_use = 1

            groups = [[] for _ in range(2**k_to_use)]
            for tag in tags:
                feature_vector = tag.id[prefix_len: prefix_len + k_to_use]
                group_index = int(feature_vector, 2)
                groups[group_index].append(tag)

            self.planned_groups = groups
            self.sub_slot_cursor = 0
            self.current_split_prefix_len = prefix_len
            self.current_k_to_use = k_to_use
            self.pending_cmd_reader_bits = CONSTANTS.READER_CMD_BASE_BITS + prefix_len + 3

            self.current_mode = 'EXECUTING_PHASE'
            return self._create_step_result('internal_op', operation_description=f"Planned DQTA Query, k={k_to_use}")

        elif self.current_mode == 'EXECUTING_PHASE':
            group = self.planned_groups[self.sub_slot_cursor]
            self.sub_slot_cursor += 1

            remaining_id_len = self.id_length - \
                self.current_split_prefix_len - self.current_k_to_use
            reader_bits_cost = self.pending_cmd_reader_bits
            self.pending_cmd_reader_bits = 0.0

            if len(group) == 0:
                self.metrics['idle_slots'] += 1
                result = self._create_step_result(
                    'idle_slot', reader_bits=reader_bits_cost)

            elif len(group) == 1:
                tag = group[0]

                perfect_response = tag.id[self.id_length - remaining_id_len:]

                noisy_response = apply_ber_noise(perfect_response, self.ber)

                if perfect_response == noisy_response:

                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] += 1
                    result = self._create_step_result('success_slot',
                                                      reader_bits=reader_bits_cost,
                                                      tag_bits=remaining_id_len,
                                                      expected_max_tag_bits=remaining_id_len)
                else:

                    self.query_stack.insert(
                        0, DQTAState(tags_to_identify=group))
                    self.metrics['collision_slots'] += 1

                    tag_bits_cost = remaining_id_len
                    result = self._create_step_result('collision_slot',
                                                      reader_bits=reader_bits_cost,
                                                      tag_bits=tag_bits_cost,
                                                      expected_max_tag_bits=remaining_id_len,
                                                      operation_description="Success slot failed due to BER")
            else:
                self.query_stack.insert(0, DQTAState(tags_to_identify=group))
                self.metrics['collision_slots'] += 1
                tag_bits_cost = len(group) * remaining_id_len
                result = self._create_step_result('collision_slot',
                                                  reader_bits=reader_bits_cost,
                                                  tag_bits=tag_bits_cost,
                                                  expected_max_tag_bits=remaining_id_len)

            if self.sub_slot_cursor == len(self.planned_groups):
                self.current_mode = 'PLANNING_PHASE'

            return result

        return self._create_step_result('internal_op', operation_description="Error: Unknown state")
