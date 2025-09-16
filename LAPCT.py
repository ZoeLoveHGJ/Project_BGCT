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
class LAPCTState:
    tags_to_identify: List[Tag] = field(default_factory=list)
    depth: int = 0


class LAPCTAlgorithm(TraditionalAlgorithmInterface):

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)
        initial_state = LAPCTState(
            tags_to_identify=self.tags_in_field, depth=0)
        self.query_stack: List[LAPCTState] = [initial_state]
        self.total_tag_count = len(tags_in_field)
        self.k_threshold_divisor = kwargs.get('k_threshold_divisor', 3.0)
        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0
        self.tag_response_counts: Dict[str, int] = {
            t.id: 0 for t in tags_in_field}

        self.ber = kwargs.get('ber', 0.0)
        self.enable_monitoring = kwargs.get(
            'enable_resource_monitoring', False)

    def _create_step_result(self, *args, **kwargs) -> AlgorithmStepResult:

        result = AlgorithmStepResult(*args, **kwargs)
        if self.enable_monitoring:
            result.internal_metrics = {'stack_depth': len(self.query_stack)}
        return result

    def is_finished(self) -> bool:
        finished = len(self.identified_tags) == self.total_tag_count
        if finished and 'avg_tag_responses' not in self.metrics:
            counts = list(self.tag_response_counts.values())
            self.metrics['avg_tag_responses'] = np.mean(
                counts) if counts else 0
        return finished

    def perform_step(self) -> AlgorithmStepResult:
        if not self.query_stack:
            return self._create_step_result(operation_type='internal_op')

        current_state = self.query_stack.pop(0)
        tags_in_this_state = [
            t for t in current_state.tags_to_identify if t.id not in self.identified_tags]

        if not tags_in_this_state:
            return self._create_step_result(operation_type='internal_op')

        for tag in tags_in_this_state:
            self.tag_response_counts[tag.id] += 1

        common_prefix, collision_positions = RfidUtils.get_collision_info(
            [t.id for t in tags_in_this_state])

        if len(tags_in_this_state) == 1:
            tag = tags_in_this_state[0]
            remaining_len = self.id_length - len(common_prefix)
            reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(common_prefix)

            perfect_response = tag.id[len(common_prefix):]
            noisy_response = apply_ber_noise(perfect_response, self.ber)

            if perfect_response == noisy_response:
                self.identified_tags.add(tag.id)
                self.metrics['success_slots'] += 1
                return self._create_step_result('success_slot', reader_bits, remaining_len, remaining_len)
            else:
                self.query_stack.insert(0, LAPCTState(
                    tags_to_identify=tags_in_this_state, depth=current_state.depth + 1))
                self.metrics['collision_slots'] += 1
                return self._create_step_result('collision_slot', reader_bits, remaining_len, remaining_len,
                                                operation_description="Success slot failed due to BER")

        num_tags = len(tags_in_this_state)
        k_threshold = 0
        if num_tags >= self.k_threshold_divisor:
            k_threshold = math.floor(
                math.log(num_tags / self.k_threshold_divisor, 4))

        use_4_way_split = (current_state.depth <=
                           k_threshold and len(collision_positions) >= 2)

        if use_4_way_split:
            self.metrics['collision_slots'] += 1
            c1, c2 = collision_positions[0], collision_positions[1]
            groups = {'00': [], '01': [], '10': [], '11': []}
            for tag in tags_in_this_state:
                groups[tag.id[c1] + tag.id[c2]].append(tag)

            total_tag_bits = 0
            bits_per_tag = self.id_length - len(common_prefix)

            for feature_bits in sorted(groups.keys()):
                group = groups[feature_bits]
                if not group:
                    self.metrics['idle_slots'] += 1
                elif len(group) == 1:
                    tag = group[0]
                    total_tag_bits += bits_per_tag

                    perfect_response = tag.id[len(common_prefix):]
                    noisy_response = apply_ber_noise(
                        perfect_response, self.ber)
                    if perfect_response == noisy_response:
                        self.identified_tags.add(tag.id)
                        self.metrics['success_slots'] += 1
                    else:
                        self.query_stack.insert(0, LAPCTState(
                            tags_to_identify=group, depth=current_state.depth + 1))
                        self.metrics['collision_slots'] += 1
                else:
                    self.query_stack.insert(0, LAPCTState(
                        tags_to_identify=group, depth=current_state.depth + 1))
                    self.metrics['collision_slots'] += 1
                    total_tag_bits += len(group) * bits_per_tag

            reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(common_prefix)
            return self._create_step_result('collision_slot', reader_bits, total_tag_bits, bits_per_tag)
        else:
            self.metrics['collision_slots'] += 1
            c1 = collision_positions[0]
            group_0 = [t for t in tags_in_this_state if t.id[c1] == '0']
            group_1 = [t for t in tags_in_this_state if t.id[c1] == '1']

            if group_1:
                self.query_stack.insert(0, LAPCTState(
                    tags_to_identify=group_1, depth=current_state.depth + 1))
            if group_0:
                self.query_stack.insert(0, LAPCTState(
                    tags_to_identify=group_0, depth=current_state.depth + 1))

            reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(common_prefix)
            expected_bits = self.id_length - len(common_prefix)
            total_tag_bits = num_tags * expected_bits
            return self._create_step_result('collision_slot', reader_bits, total_tag_bits, expected_bits)
