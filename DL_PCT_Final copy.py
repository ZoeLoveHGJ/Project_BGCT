

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set


from Framework import TraditionalAlgorithmInterface, AlgorithmStepResult, Tag, CONSTANTS
from Tool import RfidUtils


@dataclass
class DLPCTFinalState:
    """
    用于在主堆栈中存储待处理的标签集合。
    """
    tags_to_identify: List[Tag] = field(default_factory=list)


class DL_PCT_Final_Algorithm(TraditionalAlgorithmInterface):
    """
    DL-PCT 的最终版实现，通过位图预测和空间多样性选择，追求极致的性能和稳健性。
    """

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)
        self.d_max = kwargs.get('d_max', 8)
        initial_state = DLPCTFinalState(tags_to_identify=self.tags_in_field)
        self.query_stack: List[DLPCTFinalState] = [initial_state]
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

    def is_finished(self) -> bool:
        finished = len(self.identified_tags) == len(self.tags_in_field)
        if finished and 'avg_tag_responses' not in self.metrics:
            counts = list(self.tag_response_counts.values())
            if counts:
                self.metrics['avg_tag_responses'] = np.mean(counts)
                self.metrics['min_tag_responses'] = np.min(counts)
                self.metrics['max_tag_responses'] = np.max(counts)
            else:
                self.metrics.update(
                    {'avg_tag_responses': 0, 'min_tag_responses': 0, 'max_tag_responses': 0})
        return finished

    def perform_step(self) -> AlgorithmStepResult:

        if self.current_mode == 'PLANNING_PHASE':
            if not self.query_stack:
                return AlgorithmStepResult('internal_op', operation_description="Finished")

            current_state = self.query_stack.pop(0)
            tags_in_this_state = [
                t for t in current_state.tags_to_identify if t.id not in self.identified_tags]
            num_tags = len(tags_in_this_state)

            if num_tags <= 1:
                if num_tags == 1:
                    tag = tags_in_this_state[0]
                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] += 1
                    self.tag_response_counts[tag.id] += 1

                    common_prefix, _ = RfidUtils.get_collision_info(
                        [t.id for t in tags_in_this_state])
                    reader_bits = CONSTANTS.READER_CMD_BASE_BITS + \
                        len(common_prefix)
                    tag_bits = self.id_length - len(common_prefix)
                    return AlgorithmStepResult('success_slot', reader_bits, tag_bits, tag_bits)
                else:
                    return AlgorithmStepResult('internal_op', operation_description="Empty state popped")

            self.tags_for_planning = tags_in_this_state
            self.current_mode = 'AWAITING_PLANNING_DATA'

            self.metrics['collision_slots'] += 1
            for tag in self.tags_for_planning:
                self.tag_response_counts[tag.id] += 1

            tag_ids = [t.id for t in self.tags_for_planning]
            common_prefix, _ = RfidUtils.get_collision_info(tag_ids)
            prefix_len = len(common_prefix)
            remaining_len = self.id_length - prefix_len

            reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS + prefix_len

            tag_bits_cost = num_tags * remaining_len

            expected_window = remaining_len

            return AlgorithmStepResult('collision_slot',
                                       reader_bits=reader_bits_cost,
                                       tag_bits=tag_bits_cost,
                                       expected_max_tag_bits=expected_window,
                                       operation_description=f"Initial collision to get info for {num_tags} tags")

        elif self.current_mode == 'AWAITING_PLANNING_DATA':
            tags_in_this_state = self.tags_for_planning
            self.tags_for_planning = []

            tag_ids = [t.id for t in tags_in_this_state]
            common_prefix, collision_positions = RfidUtils.get_collision_info(
                tag_ids)
            d_detected = len(collision_positions)

            if d_detected == 0:
                if len(common_prefix) < self.id_length:
                    collision_positions = [len(common_prefix)]
                    d_detected = 1
                else:
                    tag = tags_in_this_state[0]
                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] += 1
                    self.tag_response_counts[tag.id] += 1
                    self.current_mode = 'PLANNING_PHASE'
                    return AlgorithmStepResult('internal_op', operation_description="Identified a duplicate tag")

            d_to_use = min(d_detected, self.d_max)

            if d_detected <= d_to_use:
                positions_to_use = collision_positions
            else:
                indices = np.linspace(0, d_detected - 1, d_to_use, dtype=int)
                positions_to_use = [collision_positions[i] for i in indices]

            num_sub_slots = 2**d_to_use
            groups = [[] for _ in range(num_sub_slots)]
            for tag in tags_in_this_state:
                feature_vector = "".join([tag.id[i] for i in positions_to_use])
                group_index = int(feature_vector, 2)
                groups[group_index].append(tag)

            self.planned_groups = groups
            self.current_d_to_use = d_to_use
            self.current_split_prefix_len = len(common_prefix)

            bits_for_d = 5
            bits_per_pos = 7
            self.pending_cmd_reader_bits = CONSTANTS.READER_CMD_BASE_BITS + \
                len(common_prefix) + bits_for_d + d_to_use * bits_per_pos

            self.current_mode = 'PREDICTION_PHASE'

            return AlgorithmStepResult('internal_op',
                                       operation_description=f"Planned QueryDL-Predict, d={d_to_use}")

        elif self.current_mode == 'PREDICTION_PHASE':
            self.non_idle_groups = [
                group for group in self.planned_groups if group]

            if not self.non_idle_groups:
                self.current_mode = 'PLANNING_PHASE'
                return AlgorithmStepResult('internal_op', operation_description="Prediction resulted in all idle slots")

            self.current_mode = 'EXECUTING_PHASE'
            self.sub_slot_cursor = 0

            bitmap_len = 2**self.current_d_to_use

            reader_bits_cost = self.pending_cmd_reader_bits
            self.pending_cmd_reader_bits = 0.0

            return AlgorithmStepResult('collision_slot',
                                       reader_bits=reader_bits_cost,
                                       tag_bits=bitmap_len,
                                       expected_max_tag_bits=bitmap_len,
                                       operation_description=f"Received prediction bitmap (len={bitmap_len})")

        elif self.current_mode == 'EXECUTING_PHASE':
            current_group = self.non_idle_groups[self.sub_slot_cursor]
            num_tags_in_group = len(current_group)
            self.sub_slot_cursor += 1

            expected_max_tag_bits = self.id_length - self.current_split_prefix_len
            result = None

            if num_tags_in_group == 1:
                tag = current_group[0]
                self.identified_tags.add(tag.id)
                self.metrics['success_slots'] += 1
                actual_tag_bits = expected_max_tag_bits
                result = AlgorithmStepResult('success_slot',
                                             tag_bits=actual_tag_bits,
                                             expected_max_tag_bits=expected_max_tag_bits)
            else:
                self.query_stack.insert(0, DLPCTFinalState(
                    tags_to_identify=current_group))
                self.metrics['collision_slots'] += 1
                total_collision_bits = num_tags_in_group * expected_max_tag_bits
                result = AlgorithmStepResult('collision_slot',
                                             tag_bits=total_collision_bits,
                                             expected_max_tag_bits=expected_max_tag_bits)

            if self.sub_slot_cursor == len(self.non_idle_groups):
                self.current_mode = 'PLANNING_PHASE'
                self.non_idle_groups = []

            return result

        return AlgorithmStepResult('internal_op', operation_description="Error: Unknown state")
