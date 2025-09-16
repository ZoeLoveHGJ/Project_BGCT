

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


from Framework import TraditionalAlgorithmInterface, AlgorithmStepResult, Tag, CONSTANTS
from Tool import RfidUtils


@dataclass
class TagState:
    """
    用于在算法内部模拟每个标签的状态。
    """
    sc: int = 0
    pointer: int = 0


class ICT_Algorithm(TraditionalAlgorithmInterface):
    """
    改进碰撞树 (Improved Collision Tree, ICT) 算法的完整复现。
    """

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)

        self.stack: List[str] = None
        self.tag_states: Dict[str, TagState] = {}
        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0

    def is_finished(self) -> bool:

        return len(self.identified_tags) == len(self.tags_in_field)

    def perform_step(self) -> AlgorithmStepResult:
        """
        算法的核心状态机，每次调用模拟一次读写器的查询-反馈周期。
        """

        if self.stack is None:

            self.stack = ['1', '0']
            self.tag_states = {t.id: TagState() for t in self.tags_in_field}

            return AlgorithmStepResult('internal_op', operation_description="初始化堆栈和标签状态")

        if not self.stack:
            return AlgorithmStepResult('internal_op', operation_description="完成")

        prefix = self.stack.pop(0)
        query_bit = prefix[-1]

        current_pointer = len(prefix) - 1

        if current_pointer >= self.id_length:

            return AlgorithmStepResult('internal_op', operation_description=f"前缀 '{prefix}' 过长，跳过")

        responders = []
        non_responding_sc0 = []

        for tag in self.tags_in_field:
            if tag.id in self.identified_tags:
                continue

            state = self.tag_states[tag.id]
            if state.sc == 0:

                state.pointer = current_pointer

                if tag.id[state.pointer] == query_bit:
                    responders.append(tag)
                else:
                    non_responding_sc0.append(tag)

        if len(responders) == 1:
            tag = responders[0]
            self.identified_tags.add(tag.id)
            self.metrics['success_slots'] += 1

            for t_id, state in self.tag_states.items():
                if t_id != tag.id and state.sc != -1:
                    state.sc -= 1
            self.tag_states[tag.id].sc = -1

            response_len = self.id_length - (current_pointer + 1)
            reader_bits = 1 + 1

            return AlgorithmStepResult(
                operation_type='success_slot',
                reader_bits=reader_bits,
                tag_bits=response_len,
                expected_max_tag_bits=response_len,
                operation_description=f"成功识别 {tag.id}"
            )

        elif len(responders) > 1:
            self.metrics['collision_slots'] += 1

            for tag in non_responding_sc0:
                self.tag_states[tag.id].sc += 1

            responses = [t.id[current_pointer + 1:] for t in responders]
            common_response_part, _ = RfidUtils.get_collision_info(responses)
            new_prefix_base = prefix + common_response_part

            self.stack.insert(0, new_prefix_base + '1')
            self.stack.insert(0, new_prefix_base + '0')

            response_len = self.id_length - (current_pointer + 1)
            reader_bits = 1 + 8
            tag_bits = len(responders) * response_len

            return AlgorithmStepResult(
                operation_type='collision_slot',
                reader_bits=reader_bits,
                tag_bits=tag_bits,
                expected_max_tag_bits=response_len,
                operation_description=f"在 {prefix} 处发生碰撞"
            )

        else:
            self.metrics['idle_slots'] += 1

            for t_id, state in self.tag_states.items():
                if state.sc != -1:
                    state.sc -= 1

            reader_bits = 1 + 1

            return AlgorithmStepResult(
                operation_type='idle_slot',
                reader_bits=reader_bits,
                tag_bits=0,
                expected_max_tag_bits=0,
                operation_description=f"在 {prefix} 处为空闲时隙"
            )
