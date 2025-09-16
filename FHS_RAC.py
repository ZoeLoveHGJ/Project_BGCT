from typing import List, Set, Dict


from Framework import TraditionalAlgorithmInterface, AlgorithmStepResult, Tag, CONSTANTS
from Tool import RfidUtils


class FHS_RAC(TraditionalAlgorithmInterface):

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field, **kwargs)

        self.query_stack: List[str] = ['']

        self.collision_factor_threshold: float = 0.75

        self.tag_response_counts: Dict[str, int] = {
            tag.id: 0 for tag in self.tags_in_field}

        self.metrics_finalized: bool = False

    def is_finished(self) -> bool:
        all_identified = len(self.identified_tags) == len(self.tags_in_field)
        finished = not self.query_stack and all_identified

        if finished and not self.metrics_finalized:
            total_responses = sum(self.tag_response_counts.values())
            num_tags = len(self.tags_in_field)
            self.metrics['avg_tag_responses'] = total_responses / \
                num_tags if num_tags > 0 else 0
            self.metrics_finalized = True

        return finished

    def perform_step(self) -> AlgorithmStepResult:
        if not self.query_stack:
            if not self.is_finished():
                print(
                    f"警告: FHS_RAC 查询栈已空，但仍有 {self.get_active_tag_count()} 个标签未识别。")
            return AlgorithmStepResult(operation_type='internal_op')

        current_prefix = self.query_stack.pop(0)

        matching_tags = [
            tag for tag in self.tags_in_field
            if tag.id not in self.identified_tags and tag.id.startswith(current_prefix)
        ]

        for tag in matching_tags:
            self.tag_response_counts[tag.id] += 1

        num_matching = len(matching_tags)
        reader_bits_sent = CONSTANTS.READER_CMD_BASE_BITS + len(current_prefix)

        current_internal_metrics = {'stack_depth': len(self.query_stack)}

        if num_matching == 0:
            self.metrics['idle_slots'] += 1
            return AlgorithmStepResult(
                operation_type='idle_slot',
                reader_bits=reader_bits_sent,
                operation_description=f"查询 '{current_prefix}...': 空闲",
                internal_metrics=current_internal_metrics
            )

        elif num_matching == 1:
            identified_tag = matching_tags[0]
            self.identified_tags.add(identified_tag.id)
            self.metrics['success_slots'] += 1

            remaining_id_len = len(identified_tag.id) - len(current_prefix)
            tag_bits_sent = CONSTANTS.RN16_RESPONSE_BITS + remaining_id_len

            return AlgorithmStepResult(
                operation_type='success_slot',
                reader_bits=reader_bits_sent,
                tag_bits=tag_bits_sent,
                expected_max_tag_bits=tag_bits_sent,
                operation_description=f"查询 '{current_prefix}...': 成功识别 {identified_tag.id[:15]}...",
                internal_metrics=current_internal_metrics
            )

        else:
            self.metrics['collision_slots'] += 1

            colliding_ids = [tag.id for tag in matching_tags]
            lcp, collision_positions = RfidUtils.get_collision_info(
                colliding_ids)

            if not collision_positions:
                identified_tag = matching_tags[0]
                self.identified_tags.add(identified_tag.id)
                self.metrics['success_slots'] += 1
                self.metrics['collision_slots'] -= 1
                remaining_id_len = len(identified_tag.id) - len(current_prefix)
                tag_bits_sent = CONSTANTS.RN16_RESPONSE_BITS + remaining_id_len
                return AlgorithmStepResult(
                    operation_type='success_slot',
                    reader_bits=reader_bits_sent,
                    tag_bits=tag_bits_sent,
                    expected_max_tag_bits=tag_bits_sent,
                    operation_description=f"查询 '{current_prefix}...': 检测到无法分裂的碰撞，强制识别一个",
                    internal_metrics=current_internal_metrics
                )

            is_highest_bit_continuous = False
            if len(collision_positions) >= 2 and collision_positions[0] + 1 == collision_positions[1]:
                is_highest_bit_continuous = True

            alpha = len(collision_positions)
            beta = len(colliding_ids[0]) - len(current_prefix)
            collision_factor = alpha / beta if beta > 0 else 0

            use_4_ary = is_highest_bit_continuous and (
                collision_factor > self.collision_factor_threshold)

            split_base = lcp

            if use_4_ary:
                new_suffixes = ['00', '01', '10', '11']
                split_len = 2
            else:
                new_suffixes = ['0', '1']
                split_len = 1

            for suffix in reversed(new_suffixes):
                self.query_stack.insert(0, split_base + suffix)

            current_internal_metrics = {'stack_depth': len(self.query_stack)}

            tag_bits_in_collision = len(
                split_base) + split_len - len(current_prefix)

            return AlgorithmStepResult(
                operation_type='collision_slot',
                reader_bits=reader_bits_sent,
                tag_bits=tag_bits_in_collision,
                expected_max_tag_bits=tag_bits_in_collision,
                operation_description=f"查询 '{current_prefix}...': {num_matching}个标签碰撞，分裂为 {'4-ary' if use_4_ary else '2-ary'}",
                internal_metrics=current_internal_metrics
            )

    def get_results(self) -> Set[str]:
        return self.identified_tags
