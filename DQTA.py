
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set

from Framework import (
    TraditionalAlgorithmInterface,
    AlgorithmStepResult,
    Tag,
    CONSTANTS
)


@dataclass
class DQTAState:
    tags_to_identify: List[Tag] = field(default_factory=list)


class DQTAAlgorithm(TraditionalAlgorithmInterface):
    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field, **kwargs)

        self.k_max = kwargs.get('k_max', 3)
        initial_state = DQTAState(tags_to_identify=self.tags_in_field)
        self.query_stack: List[DQTAState] = [initial_state]
        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0

        self.metrics['total_tag_responses'] = 0

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
        result = AlgorithmStepResult(*args, **kwargs)
        if self.enable_monitoring:
            result.internal_metrics = {'stack_depth': len(self.query_stack)}
        return result

    def _get_consecutive_collision_info(self, signal: str, start_index: int) -> int:
        if start_index >= len(signal):
            return 0

        k_detected = 0
        for i in range(start_index, len(signal)):
            if signal[i] == 'X':
                k_detected += 1
            else:
                break
        return k_detected

    def is_finished(self) -> bool:
        finished = len(self.identified_tags) == len(self.tags_in_field)

        if finished and 'avg_tag_responses' not in self.metrics:
            total_tags = len(self.tags_in_field)
            if total_tags > 0:
                self.metrics['avg_tag_responses'] = self.metrics['total_tag_responses'] / total_tags
            else:
                self.metrics['avg_tag_responses'] = 0
        return finished

    def perform_step(self) -> AlgorithmStepResult:

        if self.current_mode == 'PLANNING_PHASE':
            if not self.query_stack:
                return self._create_step_result('internal_op', operation_description="Finished")

            current_state = self.query_stack.pop(0)
            candidate_tags = [
                t for t in current_state.tags_to_identify if t.id not in self.identified_tags]

            if not candidate_tags:
                return self._create_step_result('internal_op', operation_description="Empty state popped")

            active_tags = self.channel.filter_active_tags(candidate_tags)

            if not active_tags:
                self.metrics['idle_slots'] += 1

                self.query_stack.insert(0, DQTAState(
                    tags_to_identify=candidate_tags))
                return self._create_step_result('idle_slot', reader_bits=CONSTANTS.READER_CMD_BASE_BITS)

            self.metrics['total_tag_responses'] += len(active_tags)
            for t in active_tags:
                self.tag_response_counts[t.id] += 1

            signal, _ = self.channel.resolve_collision(active_tags)

            if len(active_tags) == 1:
                tag = active_tags[0]

                if 'X' in signal:
                    self.query_stack.insert(0, DQTAState(
                        tags_to_identify=candidate_tags))
                    self.metrics['collision_slots'] += 1
                    return self._create_step_result('collision_slot', reader_bits=CONSTANTS.READER_CMD_BASE_BITS,
                                                    tag_bits=self.id_length, expected_max_tag_bits=self.id_length)

                if signal == tag.id:
                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] += 1
                    return self._create_step_result('success_slot',
                                                    reader_bits=CONSTANTS.READER_CMD_BASE_BITS,
                                                    tag_bits=self.id_length,
                                                    expected_max_tag_bits=self.id_length)
                else:

                    self.query_stack.insert(0, DQTAState(
                        tags_to_identify=candidate_tags))
                    self.metrics['collision_slots'] += 1
                    return self._create_step_result('collision_slot', reader_bits=CONSTANTS.READER_CMD_BASE_BITS,
                                                    tag_bits=self.id_length, expected_max_tag_bits=self.id_length)

            self.tags_for_planning = active_tags
            self.current_mode = 'AWAITING_PLANNING_DATA'

            try:
                prefix_len = signal.index('X')
            except ValueError:
                prefix_len = len(signal)

            if prefix_len == len(signal):
                self.query_stack.insert(
                    0, DQTAState(tags_to_identify=active_tags))
                self.metrics['collision_slots'] += 1
                return self._create_step_result('collision_slot', reader_bits=CONSTANTS.READER_CMD_BASE_BITS,
                                                tag_bits=len(active_tags)*self.id_length, expected_max_tag_bits=self.id_length)

            self.metrics['collision_slots'] += 1
            remaining_len = self.id_length - prefix_len
            reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS + prefix_len
            tag_bits_cost = len(active_tags) * remaining_len

            return self._create_step_result('collision_slot',
                                            reader_bits=reader_bits_cost,
                                            tag_bits=tag_bits_cost,
                                            expected_max_tag_bits=remaining_len,
                                            operation_description=f"Initial collision for DQTA info")

        elif self.current_mode == 'AWAITING_PLANNING_DATA':
            tags = self.tags_for_planning

            signal, _ = self.channel.resolve_collision(tags)
            try:
                prefix_len = signal.index('X')
            except ValueError:
                prefix_len = len(signal)

            k_detected = self._get_consecutive_collision_info(
                signal, prefix_len)
            k_to_use = min(k_detected, self.k_max)
            if k_to_use == 0:
                k_to_use = 1

            groups = [[] for _ in range(2**k_to_use)]
            for tag in tags:
                if prefix_len + k_to_use <= self.id_length:
                    feature_vector = tag.id[prefix_len: prefix_len + k_to_use]
                    group_index = int(feature_vector, 2)
                    groups[group_index].append(tag)
                else:
                    groups[0].append(tag)

            self.planned_groups = groups
            self.sub_slot_cursor = 0
            self.current_split_prefix_len = prefix_len
            self.current_k_to_use = k_to_use
            self.pending_cmd_reader_bits = CONSTANTS.READER_CMD_BASE_BITS + prefix_len + k_to_use

            self.current_mode = 'EXECUTING_PHASE'
            return self._create_step_result('internal_op', operation_description=f"Planned DQTA Query, k={k_to_use}")

        elif self.current_mode == 'EXECUTING_PHASE':
            candidate_group = self.planned_groups[self.sub_slot_cursor]
            self.sub_slot_cursor += 1

            active_group = self.channel.filter_active_tags(candidate_group)

            remaining_id_len = self.id_length - \
                self.current_split_prefix_len - self.current_k_to_use
            if remaining_id_len < 0:
                remaining_id_len = 0

            reader_bits_cost = self.pending_cmd_reader_bits
            self.pending_cmd_reader_bits = 0.0

            if len(active_group) > 0:
                self.metrics['total_tag_responses'] += len(active_group)
                for t in active_group:
                    self.tag_response_counts[t.id] += 1

            if len(active_group) == 0:
                self.metrics['idle_slots'] += 1
                if len(candidate_group) > 0:
                    self.query_stack.insert(0, DQTAState(
                        tags_to_identify=candidate_group))
                result = self._create_step_result(
                    'idle_slot', reader_bits=reader_bits_cost)

            elif len(active_group) == 1:
                tag = active_group[0]
                perfect_response = tag.id[self.id_length - remaining_id_len:]
                received_response, _ = self.channel.resolve_collision(
                    [tag], bit_range=(self.id_length - remaining_id_len, self.id_length))

                if perfect_response == received_response:
                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] += 1
                    result = self._create_step_result('success_slot',
                                                      reader_bits=reader_bits_cost,
                                                      tag_bits=remaining_id_len,
                                                      expected_max_tag_bits=remaining_id_len)
                else:
                    self.query_stack.insert(0, DQTAState(
                        tags_to_identify=active_group))
                    self.metrics['collision_slots'] += 1
                    result = self._create_step_result('collision_slot',
                                                      reader_bits=reader_bits_cost,
                                                      tag_bits=remaining_id_len,
                                                      expected_max_tag_bits=remaining_id_len,
                                                      operation_description="Exec success slot failed due to BER")

            else:
                self.query_stack.insert(0, DQTAState(
                    tags_to_identify=active_group))
                self.metrics['collision_slots'] += 1
                tag_bits_cost = len(active_group) * remaining_id_len
                result = self._create_step_result('collision_slot',
                                                  reader_bits=reader_bits_cost,
                                                  tag_bits=tag_bits_cost,
                                                  expected_max_tag_bits=remaining_id_len)

            if self.sub_slot_cursor == len(self.planned_groups):
                self.current_mode = 'PLANNING_PHASE'

            return result

        return self._create_step_result('internal_op', operation_description="Error: Unknown state")
