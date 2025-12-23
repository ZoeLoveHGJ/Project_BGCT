
import math
import random
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

        self.ber = kwargs.get('ber', 0.0)
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

    def _get_collision_info_cost(self, active_tags: List[Tag]) -> AlgorithmStepResult:
        self.metrics['collision_slots'] += 1
        for tag in active_tags:
            self.tag_response_counts[tag.id] += 1

        signal, _ = self.channel.resolve_collision(active_tags)
        try:
            prefix_len = signal.index('X')
        except ValueError:
            prefix_len = len(signal)

        remaining_len = self.id_length - prefix_len
        reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS + prefix_len
        tag_bits_cost = len(active_tags) * remaining_len

        return self._create_step_result('collision_slot', reader_bits=reader_bits_cost, tag_bits=tag_bits_cost,
                                        expected_max_tag_bits=remaining_len)

    def perform_step(self) -> AlgorithmStepResult:
        if self.current_mode == 'PLANNING_PHASE':
            if not self.query_stack:
                if not self.is_finished():
                    self.is_finished()
                return self._create_step_result('internal_op')

            current_state = self.query_stack.pop(0)
            candidate_tags = [
                t for t in current_state.tags_to_identify if t.id not in self.identified_tags]

            if not candidate_tags:
                return self._create_step_result('internal_op')

            active_tags = self.channel.filter_active_tags(candidate_tags)

            if not active_tags:
                self.metrics['idle_slots'] += 1
                return self._create_step_result('idle_slot', reader_bits=CONSTANTS.READER_CMD_BASE_BITS)

            if len(active_tags) == 1:
                tag = active_tags[0]
                self.tag_response_counts[tag.id] += 1
                reader_bits = CONSTANTS.READER_CMD_BASE_BITS
                tag_bits = self.id_length

                received, _ = self.channel.resolve_collision([tag])
                if received == tag.id:
                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] += 1
                    return self._create_step_result('success_slot', reader_bits, tag_bits, tag_bits)
                else:
                    self.query_stack.insert(0, BGVCTState(
                        tags_to_identify=candidate_tags))
                    self.metrics['collision_slots'] += 1
                    return self._create_step_result('collision_slot', reader_bits, tag_bits, tag_bits,
                                                    operation_description="Failed due to BER")

            self.tags_for_planning = active_tags
            self.current_mode = 'AWAITING_PLANNING_DATA'
            return self._get_collision_info_cost(self.tags_for_planning)

        elif self.current_mode == 'AWAITING_PLANNING_DATA':

            signal, collision_positions = self.channel.resolve_collision(
                self.tags_for_planning)
            d_detected = len(collision_positions)

            if d_detected == 0:

                self.current_mode = 'PLANNING_PHASE'
                return self._create_step_result('internal_op')

            d_to_use = min(d_detected, self.d_max)
            indices = np.linspace(0, d_detected - 1, d_to_use, dtype=int)
            positions_to_use = [collision_positions[i] for i in indices]

            groups = [[] for _ in range(2**d_to_use)]
            for tag in self.tags_for_planning:

                feature_bits = [tag.id[i] for i in positions_to_use]
                groups[int("".join(feature_bits), 2)].append(tag)

            self.planned_groups = groups
            self.current_d_to_use = d_to_use
            try:
                self.current_split_prefix_len = signal.index('X')
            except ValueError:
                self.current_split_prefix_len = len(signal)

            self.pending_cmd_reader_bits = CONSTANTS.READER_CMD_BASE_BITS + \
                self.current_split_prefix_len + 5 + d_to_use * 7
            self.current_mode = 'PREDICTION_PHASE'
            return self._create_step_result('internal_op')

        elif self.current_mode == 'PREDICTION_PHASE':
            self.non_idle_groups = [g for g in self.planned_groups if g]
            if not self.non_idle_groups:
                self.current_mode = 'PLANNING_PHASE'
                return self._create_step_result('internal_op')

            self.current_mode = 'EXECUTING_PHASE'
            self.sub_slot_cursor = 0
            bitmap_len = 2**self.current_d_to_use
            reader_bits = self.pending_cmd_reader_bits
            self.pending_cmd_reader_bits = 0.0

            num_responding_tags = sum(len(g) for g in self.non_idle_groups)
            return self._create_step_result('collision_slot', reader_bits=reader_bits,
                                            tag_bits=num_responding_tags, expected_max_tag_bits=bitmap_len)

        elif self.current_mode == 'EXECUTING_PHASE':
            candidate_group = self.non_idle_groups[self.sub_slot_cursor]
            self.sub_slot_cursor += 1

            active_group = self.channel.filter_active_tags(candidate_group)
            result = None

            if len(active_group) == 1:
                tag = active_group[0]
                self.tag_response_counts[tag.id] += 1
                reader_bits_cost = CONSTANTS.QUERYREP_CMD_BITS
                remaining_len = self.id_length - self.current_split_prefix_len

                perfect_frag = tag.id[self.current_split_prefix_len:]
                received_frag, _ = self.channel.resolve_collision(
                    [tag], bit_range=(self.current_split_prefix_len, self.id_length))

                if perfect_frag == received_frag:
                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] += 1
                    result = self._create_step_result('success_slot', reader_bits=reader_bits_cost,
                                                      tag_bits=remaining_len, expected_max_tag_bits=remaining_len)
                else:
                    self.query_stack.insert(0, BGVCTState(
                        tags_to_identify=candidate_group))
                    self.metrics['collision_slots'] += 1
                    result = self._create_step_result('collision_slot', reader_bits=reader_bits_cost,
                                                      tag_bits=remaining_len, expected_max_tag_bits=remaining_len,
                                                      operation_description="BER Fail")
            elif len(active_group) == 0:

                self.metrics['idle_slots'] += 1
                self.query_stack.insert(0, BGVCTState(
                    tags_to_identify=candidate_group))
                result = self._create_step_result(
                    'idle_slot', reader_bits=CONSTANTS.QUERYREP_CMD_BITS)
            else:
                self.query_stack.insert(0, BGVCTState(
                    tags_to_identify=active_group))
                self.metrics['collision_slots'] += 1
                expected_bits = self.id_length - self.current_split_prefix_len
                reader_bits_cost = CONSTANTS.QUERYREP_CMD_BITS
                result = self._create_step_result('collision_slot', reader_bits=reader_bits_cost,
                                                  tag_bits=len(
                                                      active_group) * expected_bits,
                                                  expected_max_tag_bits=expected_bits)

            if self.sub_slot_cursor >= len(self.non_idle_groups):
                self.current_mode = 'PLANNING_PHASE'
            return result

        return self._create_step_result('internal_op')


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
                return self._create_step_result('internal_op')

            current_state = self.query_stack.pop(0)
            candidate_tags = [
                t for t in current_state.tags_to_identify if t.id not in self.identified_tags]

            if not candidate_tags:
                return self._create_step_result('internal_op')

            active_tags = self.channel.filter_active_tags(candidate_tags)

            if not active_tags:
                self.metrics['idle_slots'] += 1
                return self._create_step_result('idle_slot', reader_bits=CONSTANTS.READER_CMD_BASE_BITS)

            if len(active_tags) == 1:

                tag = active_tags[0]
                self.tag_response_counts[tag.id] += 1
                received, _ = self.channel.resolve_collision([tag])
                if received == tag.id:
                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] += 1
                    return self._create_step_result('success_slot', CONSTANTS.READER_CMD_BASE_BITS, self.id_length, self.id_length)
                else:
                    self.query_stack.insert(
                        0, BGVCTState(tags_to_identify=active_tags))
                    self.metrics['collision_slots'] += 1
                    return self._create_step_result('collision_slot', CONSTANTS.READER_CMD_BASE_BITS, self.id_length, self.id_length)

            self.tags_for_planning = active_tags
            signal, _ = self.channel.resolve_collision(self.tags_for_planning)
            try:
                self.probe_offset = signal.index('X')
            except ValueError:
                self.probe_offset = len(signal)
            self.accumulated_positions = []
            self.current_mode = 'PROGRESSIVE_PROBE'
            return self._create_step_result('internal_op')

        elif self.current_mode == 'PROGRESSIVE_PROBE':
            if self.probe_offset >= self.id_length or len(self.accumulated_positions) >= self.d_target:
                return self._process_accumulated_info()

            start, end = self.probe_offset, min(
                self.probe_offset + self.chunk_size, self.id_length)
            current_chunk_len = end - start

            if current_chunk_len <= 0:
                return self._process_accumulated_info()

            _, chunk_indices = self.channel.resolve_collision(
                self.tags_for_planning, bit_range=(start, end))

            for rel_idx in chunk_indices:
                abs_idx = start + rel_idx
                if abs_idx not in self.accumulated_positions:
                    self.accumulated_positions.append(abs_idx)

            self.probe_offset = end
            self.metrics['collision_slots'] += 1
            for tag in self.tags_for_planning:
                self.tag_response_counts[tag.id] += 1

            reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS + start
            tag_bits_cost = len(self.tags_for_planning) * current_chunk_len

            return self._create_step_result('collision_slot', reader_bits=reader_bits_cost, tag_bits=tag_bits_cost,
                                            expected_max_tag_bits=current_chunk_len)

        return self._create_step_result('internal_op')

    def _process_accumulated_info(self) -> AlgorithmStepResult:
        """
        标准版策略: Earliest-Position-First (最早位置优先)。
        选择前 d_target 个发现的碰撞位。
        """
        collision_positions = self.accumulated_positions
        d_detected = len(collision_positions)

        if d_detected == 0:
            self.current_mode = 'PLANNING_PHASE'
            return self._create_step_result('internal_op')

        d_to_use = min(d_detected, self.d_target)

        positions_to_use = collision_positions[:d_to_use]

        groups = [[] for _ in range(2**d_to_use)]
        for tag in self.tags_for_planning:
            feature_bits = [tag.id[i] for i in positions_to_use]
            groups[int("".join(feature_bits), 2)].append(tag)

        self.planned_groups = groups
        self.current_d_to_use = d_to_use

        signal, _ = self.channel.resolve_collision(self.tags_for_planning)
        try:
            self.current_split_prefix_len = signal.index('X')
        except ValueError:
            self.current_split_prefix_len = len(signal)

        self.pending_cmd_reader_bits = CONSTANTS.READER_CMD_BASE_BITS + \
            self.current_split_prefix_len + 5 + d_to_use * 7
        self.current_mode = 'PREDICTION_PHASE'

        return self._create_step_result('internal_op')


class BGCT_RandomSelection(BGCT):
    def _process_accumulated_info(self) -> AlgorithmStepResult:

        collision_positions = self.accumulated_positions
        d_detected = len(collision_positions)

        if d_detected == 0:
            self.current_mode = 'PLANNING_PHASE'
            return self._create_step_result('internal_op')

        d_to_use = min(d_detected, self.d_target)

        if d_detected > 0:
            selected_positions = random.sample(collision_positions, d_to_use)
            positions_to_use = sorted(selected_positions)
        else:
            positions_to_use = []

        groups = [[] for _ in range(2**d_to_use)]
        for tag in self.tags_for_planning:
            feature_bits = [tag.id[i] for i in positions_to_use]
            groups[int("".join(feature_bits), 2)].append(tag)

        self.planned_groups = groups
        self.current_d_to_use = d_to_use

        signal, _ = self.channel.resolve_collision(self.tags_for_planning)
        try:
            self.current_split_prefix_len = signal.index('X')
        except ValueError:
            self.current_split_prefix_len = len(signal)

        self.pending_cmd_reader_bits = CONSTANTS.READER_CMD_BASE_BITS + \
            self.current_split_prefix_len + 5 + d_to_use * 7
        self.current_mode = 'PREDICTION_PHASE'

        return self._create_step_result('internal_op')
