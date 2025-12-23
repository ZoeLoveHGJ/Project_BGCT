from typing import List, Set, Dict


from Framework import TraditionalAlgorithmInterface, AlgorithmStepResult, Tag, CONSTANTS


class FHS_RAC(TraditionalAlgorithmInterface):
    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field, **kwargs)

        self.query_stack: List[str] = ['']

        self.collision_factor_threshold: float = 0.75

        self.metrics['total_tag_responses'] = 0

    def is_finished(self) -> bool:
        """
        判断算法是否结束：栈为空且所有标签被识别
        """
        return len(self.query_stack) == 0 and len(self.identified_tags) == len(self.tags_in_field)

    def perform_step(self) -> AlgorithmStepResult:
        """
        执行一个时隙的仿真步骤
        """

        if not self.query_stack:
            return AlgorithmStepResult(operation_type='internal_op')

        current_prefix = self.query_stack.pop(0)
        reader_bits_sent = CONSTANTS.READER_CMD_BASE_BITS + len(current_prefix)

        candidates = [
            tag for tag in self.tags_in_field
            if tag.id not in self.identified_tags and tag.id.startswith(current_prefix)
        ]

        active_tags = self.channel.filter_active_tags(candidates)

        self.metrics['total_tag_responses'] += len(active_tags)

        current_internal_metrics = {'stack_depth': len(self.query_stack)}

        start_pos = len(current_prefix)

        signal_str, collision_indices = self.channel.resolve_collision(
            active_tags, bit_range=(start_pos, None))

        if not active_tags:
            self.metrics['idle_slots'] += 1
            return AlgorithmStepResult(
                operation_type='idle_slot',
                reader_bits=reader_bits_sent,
                operation_description=f"Query '{current_prefix}': Idle",
                internal_metrics=current_internal_metrics
            )

        elif len(collision_indices) == 0:
            self.metrics['success_slots'] += 1

            if active_tags:
                identified_tag = active_tags[0]
                self.identified_tags.add(identified_tag.id)

                remaining_id_len = len(signal_str)
                tag_bits_sent = CONSTANTS.RN16_RESPONSE_BITS + remaining_id_len

                return AlgorithmStepResult(
                    operation_type='success_slot',
                    reader_bits=reader_bits_sent,
                    tag_bits=tag_bits_sent,
                    expected_max_tag_bits=tag_bits_sent,
                    operation_description=f"Query '{current_prefix}': Success ({identified_tag.id[-4:]})",
                    internal_metrics=current_internal_metrics
                )
            else:

                return AlgorithmStepResult(operation_type='internal_op')

        else:
            self.metrics['collision_slots'] += 1

            alpha = len(collision_indices)

            beta = len(signal_str)

            collision_factor = alpha / beta if beta > 0 else 0

            is_highest_bit_continuous = False
            if len(collision_indices) >= 2:

                if collision_indices[1] == collision_indices[0] + 1:
                    is_highest_bit_continuous = True

            use_4_ary = is_highest_bit_continuous and (
                collision_factor > self.collision_factor_threshold)

            split_base = current_prefix

            if use_4_ary:
                new_suffixes = ['00', '01', '10', '11']
                strategy_name = "4-ary"
            else:
                new_suffixes = ['0', '1']
                strategy_name = "2-ary"

            for suffix in reversed(new_suffixes):
                self.query_stack.insert(0, split_base + suffix)

            current_internal_metrics['stack_depth'] = len(self.query_stack)

            tag_bits_in_collision = len(signal_str)

            return AlgorithmStepResult(
                operation_type='collision_slot',
                reader_bits=reader_bits_sent,
                tag_bits=tag_bits_in_collision,
                expected_max_tag_bits=tag_bits_in_collision,
                operation_description=f"Query '{current_prefix}': Coll(μ={collision_factor:.2f}, {strategy_name})",
                internal_metrics=current_internal_metrics
            )
