import math
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


from Framework import (
    TraditionalAlgorithmInterface,
    AlgorithmStepResult,
    Tag,
    CONSTANTS
)


@dataclass
class EAQState:
    """EAQ 任务状态: 保存当前查询前缀及对应的标签子集"""
    prefix: str = ""
    tags_in_subset: List[Tag] = field(default_factory=list)


class EAQCBBAlgorithm(TraditionalAlgorithmInterface):

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field, **kwargs)

        self.M_tpe = kwargs.get('M_tpe', 96)
        self.b_l_max = kwargs.get('b_l', 3)

        self.state = 'TPE_PHASE'
        self.s_opt = 1
        self.partition_queue: List[List[Tag]] = []
        self.task_stack: List[EAQState] = []

        self.total_tag_count = len(tags_in_field)
        self.id_length = len(tags_in_field[0].id) if tags_in_field else 96

        self.metrics['total_tag_responses'] = 0

        self.tag_response_counts: Dict[str, int] = {
            t.id: 0 for t in tags_in_field}

        self.enable_monitoring = kwargs.get(
            'enable_resource_monitoring', False)

    def _create_step_result(self, *args, **kwargs) -> AlgorithmStepResult:
        """辅助函数：封装步骤结果并注入监控指标"""
        result = AlgorithmStepResult(*args, **kwargs)
        if self.enable_monitoring:

            depth = len(self.task_stack) + len(self.partition_queue)
            result.internal_metrics = {'stack_depth': depth}
        return result

    def is_finished(self) -> bool:
        """判断算法是否结束"""
        finished = len(self.identified_tags) == self.total_tag_count
        if finished:
            return True

        if not self.task_stack and not self.partition_queue:
            remaining = [
                t for t in self.tags_in_field if t.id not in self.identified_tags]
            if remaining:

                self.task_stack.append(
                    EAQState(prefix="", tags_in_subset=remaining))
                return False
        return False

    def perform_step(self) -> AlgorithmStepResult:
        """主状态机执行一步"""

        if self.state == 'TPE_PHASE':
            active_tags = self.channel.filter_active_tags(self.tags_in_field)

            if not active_tags:

                self.s_opt = 1
                self.state = 'PARTITIONING'
                return self._create_step_result('internal_op', operation_description="TPE: No tags")

            tpe_responses = []
            for tag in active_tags:

                self.tag_response_counts[tag.id] += 1
                self.metrics['total_tag_responses'] += 1

                bits = ['1'] * self.M_tpe
                idx = random.randint(0, self.M_tpe - 1)
                bits[idx] = '0'
                tpe_responses.append("".join(bits))

            mixed_signal, _ = self.channel.resolve_collision_raw_strings(
                tpe_responses)

            c_u = 0
            for char in mixed_signal:
                if char != 'X':
                    c_u += 1

            try:
                if c_u == 0:
                    c_u = 0.001
                term1 = math.log(c_u / self.M_tpe)
                term2 = math.log(1 - 1.0 / self.M_tpe)
                n_hat = math.ceil(term1 / term2)
            except:
                n_hat = len(active_tags)

            n_hat = max(1, int(n_hat))

            self.s_opt = max(1, int(0.25 * n_hat))

            reader_bits = 8

            total_tag_bits = len(active_tags) * self.M_tpe

            t_reader = reader_bits / CONSTANTS.READER_BITS_PER_US
            t_tag = self.M_tpe / CONSTANTS.TAG_BITS_PER_US
            time_us = t_reader + CONSTANTS.T1_US + t_tag + CONSTANTS.T2_MIN_US

            self.state = 'PARTITIONING'
            return self._create_step_result('internal_op', reader_bits, total_tag_bits,
                                            override_time_us=time_us, operation_description=f"TPE: est={n_hat}, S={self.s_opt}")

        if self.state == 'PARTITIONING':

            partitions = [[] for _ in range(self.s_opt)]

            for tag in self.tags_in_field:
                slot_idx = random.randint(0, self.s_opt - 1)
                partitions[slot_idx].append(tag)

            for p_tags in partitions:
                if not p_tags:

                    self.metrics['idle_slots'] += 1
                else:
                    self.partition_queue.append(p_tags)

            reader_bits = 64
            t_reader = reader_bits / CONSTANTS.READER_BITS_PER_US
            time_us = t_reader + CONSTANTS.T1_US

            self.state = 'PROCESSING'
            return self._create_step_result('internal_op', reader_bits, 0,
                                            override_time_us=time_us, operation_description="Partitioning")

        if self.state == 'PROCESSING':

            if not self.task_stack:
                if self.partition_queue:
                    next_group = self.partition_queue.pop(0)
                    self.task_stack.append(
                        EAQState(prefix="", tags_in_subset=next_group))
                else:

                    return self._create_step_result('internal_op')

            current_state = self.task_stack.pop(0)
            prefix = current_state.prefix

            candidates = [
                t for t in current_state.tags_in_subset if t.id not in self.identified_tags]

            active_tags = self.channel.filter_active_tags(candidates)

            for t in active_tags:
                self.tag_response_counts[t.id] += 1
                self.metrics['total_tag_responses'] += 1

            reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(prefix)

            if not active_tags:
                self.metrics['idle_slots'] += 1
                return self._create_step_result('idle_slot', reader_bits, 0, 0)

            payloads = [t.id[len(prefix):] for t in active_tags]

            if payloads and len(payloads[0]) == 0:
                for t in active_tags:
                    self.identified_tags.add(t.id)
                self.metrics['success_slots'] += 1
                return self._create_step_result('success_slot', reader_bits, 0, 0)

            observed_signal, collision_indices = self.channel.resolve_collision_raw_strings(
                payloads)
            total_tag_bits = sum(len(p) for p in payloads)
            response_len = len(payloads[0])

            if not collision_indices:
                matched_suffix = observed_signal

                full_id_candidates = [
                    t for t in active_tags if t.id[len(prefix):] == matched_suffix]

                if len(full_id_candidates) == 1:
                    self.identified_tags.add(full_id_candidates[0].id)
                    self.metrics['success_slots'] += 1
                    return self._create_step_result('success_slot', reader_bits, total_tag_bits, response_len)
                else:

                    self.metrics['collision_slots'] += 1
                    self.task_stack.insert(0, current_state)
                    return self._create_step_result('collision_slot', reader_bits, total_tag_bits, response_len,
                                                    operation_description="Hidden Collision")

            else:

                self.metrics['collision_slots'] += 1

                b_i = collision_indices[0]
                b_l = 1
                for k in range(1, self.b_l_max):
                    if (b_i + k) in collision_indices:
                        b_l += 1
                    else:
                        break

                t_query = reader_bits / CONSTANTS.READER_BITS_PER_US
                t_id = response_len / CONSTANTS.TAG_BITS_PER_US
                time_base = t_query + CONSTANTS.T1_US + t_id + CONSTANTS.T2_MIN_US

                bmreq_bits = 105

                mr_len = 2 ** b_l

                t_bmreq = bmreq_bits / CONSTANTS.READER_BITS_PER_US
                t_mr = mr_len / CONSTANTS.TAG_BITS_PER_US
                time_extra = t_bmreq + CONSTANTS.T1_US + t_mr + CONSTANTS.T2_MIN_US

                total_time_us = time_base + time_extra

                mr_responses = []
                groups = {i: [] for i in range(mr_len)}

                for tag in active_tags:

                    start = len(prefix) + b_i
                    end = start + b_l
                    if start >= len(tag.id):
                        continue

                    self.tag_response_counts[tag.id] += 1
                    self.metrics['total_tag_responses'] += 1

                    block_bits = tag.id[start: min(end, len(tag.id))]
                    if len(block_bits) < b_l:
                        block_bits = block_bits.ljust(b_l, '0')
                    val = int(block_bits, 2)

                    groups[val].append(tag)

                    mr_list = ['0'] * mr_len
                    mr_list[val] = '1'
                    mr_responses.append("".join(mr_list))

                mixed_mr, _ = self.channel.resolve_collision_raw_strings(
                    mr_responses)
                active_indices = []
                for idx, char in enumerate(mixed_mr):
                    if char == '1' or char == 'X':
                        active_indices.append(idx)

                if not active_indices:

                    self.task_stack.insert(0, current_state)
                    return self._create_step_result('collision_slot', reader_bits, total_tag_bits, response_len,
                                                    override_time_us=total_time_us, operation_description="Bitmap Error")

                uncollided_segment = observed_signal[:b_i].replace('X', '0')

                for idx in reversed(active_indices):

                    block_suffix = format(idx, f'0{b_l}b')
                    new_prefix = prefix + uncollided_segment + block_suffix
                    self.task_stack.insert(0, EAQState(
                        prefix=new_prefix, tags_in_subset=groups[idx]))

                total_reader_bits_step = reader_bits + bmreq_bits

                total_tag_bits_step = total_tag_bits + \
                    (len(active_tags) * mr_len)

                return self._create_step_result('collision_slot', total_reader_bits_step, total_tag_bits_step, response_len,
                                                override_time_us=total_time_us, operation_description=f"QTCBB: Split {len(active_indices)}")

        return self._create_step_result('internal_op')
