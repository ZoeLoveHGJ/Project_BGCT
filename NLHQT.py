
from typing import List
from Framework import TraditionalAlgorithmInterface, AlgorithmStepResult, CONSTANTS, Tag


class NLHQTAlgorithm(TraditionalAlgorithmInterface):

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field, **kwargs)

        self.n_way = kwargs.get('n_way', 2)

        self.stack = [""]
        self.current_prefix = ""

        self.metrics['total_tag_responses'] = 0

    def is_finished(self) -> bool:
        return len(self.stack) == 0

    def perform_step(self) -> AlgorithmStepResult:
        if self.is_finished():
            return AlgorithmStepResult(operation_type='internal_op')

        prefix = self.stack.pop()
        self.current_prefix = prefix

        candidates = [t for t in self.tags_in_field if t.id.startswith(prefix)]

        active_tags = self.channel.filter_active_tags(candidates)

        num_active = len(active_tags)
        self.metrics['total_tag_responses'] += num_active

        reader_bits_query = CONSTANTS.QUERY_CMD_BITS + len(prefix)
        tag_response_len = max(0, CONSTANTS.EPC_CODE_BITS - len(prefix))

        response_str, collision_indices = self.channel.resolve_collision(
            active_tags,
            bit_range=(len(prefix), CONSTANTS.EPC_CODE_BITS)
        )

        current_stack_depth = len(self.stack)

        if not active_tags:
            self.metrics['idle_slots'] += 1

            return AlgorithmStepResult(
                operation_type='idle_slot',
                reader_bits=reader_bits_query,
                tag_bits=0,
                operation_description=f"Idle: {prefix}",

                internal_metrics={'stack_depth': current_stack_depth}
            )

        elif not collision_indices:
            self.metrics['success_slots'] += 1
            for tag in active_tags:
                self.identified_tags.add(tag.id)

            return AlgorithmStepResult(
                operation_type='success_slot',
                reader_bits=reader_bits_query,
                tag_bits=tag_response_len,
                expected_max_tag_bits=tag_response_len,
                operation_description=f"Success: {prefix}",
                internal_metrics={'stack_depth': current_stack_depth}
            )

        else:
            self.metrics['collision_slots'] += 1

            self.metrics['total_tag_responses'] += num_active

            reader_bits_pred = CONSTANTS.QUERYREP_CMD_BITS
            pred_response_len = 2 ** self.n_way

            prediction_vectors = []
            for tag in active_tags:
                start_idx = len(prefix)
                end_idx = start_idx + self.n_way
                if start_idx >= CONSTANTS.EPC_CODE_BITS:
                    continue
                next_bits = tag.id[start_idx: end_idx]
                if len(next_bits) < self.n_way:
                    next_bits = next_bits.ljust(self.n_way, '0')

                val = int(next_bits, 2)
                vec = ['0'] * pred_response_len
                vec[val] = '1'
                prediction_vectors.append("".join(vec))

            mixed_pred_signal, _ = self.channel.resolve_collision_raw_strings(
                prediction_vectors)

            active_branches = []

            for i in range(pred_response_len - 1, -1, -1):

                if i < len(mixed_pred_signal) and mixed_pred_signal[i] in ['1', 'X']:
                    suffix = format(i, f'0{self.n_way}b')
                    self.stack.append(prefix + suffix)
                    active_branches.append(suffix)

            peak_stack_depth_now = len(self.stack)

            total_reader_bits = reader_bits_query + reader_bits_pred
            total_tag_bits = tag_response_len + pred_response_len

            t_r1 = (reader_bits_query / CONSTANTS.READER_BITS_PER_US) + \
                CONSTANTS.T1_US + \
                   (tag_response_len / CONSTANTS.TAG_BITS_PER_US) + \
                CONSTANTS.T2_MIN_US

            t_r2 = (reader_bits_pred / CONSTANTS.READER_BITS_PER_US) + \
                CONSTANTS.T1_US + \
                   (pred_response_len / CONSTANTS.TAG_BITS_PER_US) + \
                CONSTANTS.T2_MIN_US

            total_precise_time = t_r1 + t_r2

            return AlgorithmStepResult(
                operation_type='collision_slot',
                reader_bits=total_reader_bits,
                tag_bits=total_tag_bits,
                expected_max_tag_bits=tag_response_len,
                override_time_us=total_precise_time,
                operation_description=f"Collision: {prefix} -> Pred Active: {active_branches}",

                internal_metrics={'stack_depth': peak_stack_depth_now}
            )
