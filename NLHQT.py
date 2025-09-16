
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


from Framework import (
    TraditionalAlgorithmInterface,
    AlgorithmStepResult,
    Tag,
    CONSTANTS,
    apply_ber_noise
)
from Tool import RfidUtils


@dataclass
class NLHQTState:
    prefix: str = ""

    tags_in_subset: List[Tag] = field(default_factory=list)


class NLHQTAlgorithm(TraditionalAlgorithmInterface):

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)
        self.n_way = kwargs.get('n_way', 2)
        initial_state = NLHQTState(
            prefix="", tags_in_subset=self.tags_in_field)
        self.query_stack: List[NLHQTState] = [initial_state]
        self.is_locked = False
        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0
        self.tag_response_counts: Dict[str, int] = {
            t.id: 0 for t in tags_in_field}

        self.ber = kwargs.get('ber', 0.0)
        self.enable_monitoring = kwargs.get(
            'enable_resource_monitoring', False)

        self._global_short_id_map: Dict[str, str] = {}
        self._locked_collision_positions: List[int] = []

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

    def perform_step(self) -> AlgorithmStepResult:
        if not self.query_stack:
            return self._create_step_result(operation_type='internal_op')

        current_state = self.query_stack.pop(0)
        tags_in_this_state = [
            t for t in current_state.tags_in_subset if t.id not in self.identified_tags]

        if not tags_in_this_state:
            return self._create_step_result(operation_type='internal_op')

        if not self.is_locked:
            for tag in self.tags_in_field:
                self.tag_response_counts[tag.id] += 1

            if len(self.tags_in_field) == 1:
                tag = self.tags_in_field[0]
                perfect_response = tag.id
                noisy_response = apply_ber_noise(perfect_response, self.ber)
                if perfect_response == noisy_response:
                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] += 1
                    return self._create_step_result('success_slot', CONSTANTS.READER_CMD_BASE_BITS, self.id_length, self.id_length)

            self.is_locked = True
            self.metrics['collision_slots'] += 1
            tag_ids = [t.id for t in self.tags_in_field]

            common_prefix, collision_positions = RfidUtils.get_collision_info(
                tag_ids)
            self._locked_collision_positions = collision_positions

            self._global_short_id_map = {t.id: "".join(
                [t.id[i] for i in collision_positions]) for t in self.tags_in_field}

            self.query_stack.insert(0, NLHQTState(
                prefix="", tags_in_subset=self.tags_in_field))

            reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS + \
                len(common_prefix)
            tag_bits_cost = len(self.tags_in_field) * \
                (self.id_length - len(common_prefix))
            expected_window = self.id_length - len(common_prefix)

            return self._create_step_result('collision_slot', reader_bits_cost, tag_bits_cost, expected_window,
                                            operation_description="Global Lock-in Phase")

        prefix = current_state.prefix
        self.metrics['collision_slots'] += 1

        reader_bits_predict = CONSTANTS.READER_CMD_BASE_BITS + \
            len(prefix) + self.n_way

        short_id_len = len(self._locked_collision_positions)
        tags_matching_prefix = [t for t in tags_in_this_state if self._global_short_id_map.get(
            t.id, '').startswith(prefix)]

        occupancy_bitmap = 0
        if len(prefix) + self.n_way <= short_id_len:
            for tag in tags_matching_prefix:
                self.tag_response_counts[tag.id] += 1
                short_id = self._global_short_id_map.get(tag.id)
                next_bits = short_id[len(prefix): len(prefix) + self.n_way]
                occupancy_bitmap |= (1 << int(next_bits, 2))

        num_non_idle_children = bin(occupancy_bitmap).count('1')

        tag_bits_from_bitmap = len(tags_matching_prefix) * (2**self.n_way)
        tag_bits_from_success = 0

        if num_non_idle_children == 1:
            child_index = int(math.log2(occupancy_bitmap))
            child_suffix = format(child_index, f'0{self.n_way}b')
            new_prefix = prefix + child_suffix
            tags_in_child = [t for t in tags_matching_prefix if self._global_short_id_map.get(
                t.id, '').startswith(new_prefix)]

            if len(tags_in_child) == 1:
                tag = tags_in_child[0]
                remaining_short_id = self._global_short_id_map.get(tag.id)[
                    len(new_prefix):]
                noisy_response = apply_ber_noise(remaining_short_id, self.ber)

                if remaining_short_id == noisy_response:
                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] += 1

                    tag_bits_from_success = len(remaining_short_id)
                else:
                    self.query_stack.insert(0, NLHQTState(
                        prefix=new_prefix, tags_in_subset=tags_in_child))
                    self.metrics['collision_slots'] += 1
            else:
                self.query_stack.insert(0, NLHQTState(
                    prefix=new_prefix, tags_in_subset=tags_in_child))
                self.metrics['collision_slots'] += 1

        elif num_non_idle_children > 1:
            for i in range(2**self.n_way - 1, -1, -1):
                if (occupancy_bitmap >> i) & 1:
                    child_suffix = format(i, f'0{self.n_way}b')
                    new_prefix = prefix + child_suffix
                    tags_in_child = [t for t in tags_matching_prefix if self._global_short_id_map.get(
                        t.id, '').startswith(new_prefix)]
                    self.query_stack.insert(0, NLHQTState(
                        prefix=new_prefix, tags_in_subset=tags_in_child))

        total_tag_bits = tag_bits_from_bitmap + tag_bits_from_success

        expected_window = max(2**self.n_way, short_id_len - len(prefix))

        return self._create_step_result('collision_slot', reader_bits_predict, total_tag_bits, expected_window)
