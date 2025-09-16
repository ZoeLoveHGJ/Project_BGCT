
import math
import random
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
class EAQTask:
    """用于在主堆栈中存储待处理的标签集合"""
    tags_to_process: List[Tag]


class EAQCBBAlgorithm(TraditionalAlgorithmInterface):
    """EAQ-CBB 算法的复现实现 - V2 适配版"""

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)

        self.s_opt_ratio = kwargs.get('s_opt_ratio', 0.25)
        self.b_l = kwargs.get('b_l', 3)

        self.current_mode = 'INITIAL_PARTITIONING'
        self.task_stack: List[EAQTask] = []

        self.qtcbb_task: EAQTask = None
        self.qtcbb_prefix = ""
        self.qtcbb_b_i = 0
        self.qtcbb_b_l = 0

        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0
        self.tag_response_counts: Dict[str, int] = {
            t.id: 0 for t in tags_in_field}

        self.ber = kwargs.get('ber', 0.0)
        self.enable_monitoring = kwargs.get(
            'enable_resource_monitoring', False)

    def _create_step_result(self, *args, **kwargs) -> AlgorithmStepResult:
        """
        V2 新增: 辅助函数，用于统一创建 AlgorithmStepResult 对象。
        如果资源监控开启，此函数会自动附加内部状态指标。
        """
        result = AlgorithmStepResult(*args, **kwargs)
        if self.enable_monitoring:
            result.internal_metrics = {'stack_depth': len(self.task_stack)}
        return result

    def is_finished(self) -> bool:
        finished = len(self.identified_tags) == len(self.tags_in_field)
        if finished and 'avg_tag_responses' not in self.metrics:
            counts = list(self.tag_response_counts.values())
            self.metrics['avg_tag_responses'] = np.mean(
                counts) if counts else 0
        return finished

    def _get_collided_block_info(self, tag_ids: List[str], prefix: str) -> Tuple[int, int]:
        """获取第一个碰撞位(b_i)和连续碰撞块长度(b_l)"""
        prefix_len = len(prefix)
        if len(tag_ids) <= 1:
            return -1, 0
        id_len = len(tag_ids[0])
        b_i = -1
        for i in range(prefix_len, id_len):
            if len(set(tag[i] for tag in tag_ids)) > 1:
                b_i = i
                break
        if b_i == -1:
            return -1, 0
        b_l = 1
        for i in range(b_i + 1, id_len):
            if len(set(tag[i] for tag in tag_ids)) > 1:
                b_l += 1
            else:
                break
        return b_i, b_l

    def perform_step(self) -> AlgorithmStepResult:
        if self.is_finished():
            return self._create_step_result('internal_op', operation_description="All tags identified.")

        if self.current_mode == 'INITIAL_PARTITIONING':
            n_hat = len(self.tags_in_field)
            s_opt = max(1, math.ceil(self.s_opt_ratio * n_hat))
            partitions = [[] for _ in range(s_opt)]
            for tag in self.tags_in_field:
                partitions[random.randint(0, s_opt - 1)].append(tag)

            for p in reversed(partitions):
                if p:
                    self.task_stack.append(EAQTask(tags_to_process=p))

            self.current_mode = 'PROCESSING_PARTITIONS'
            return self._create_step_result('internal_op', operation_description=f"Partitioned into {s_opt} groups.")

        elif self.current_mode == 'PROCESSING_PARTITIONS':
            if not self.task_stack:
                return self._create_step_result('internal_op', operation_description="All tasks finished.")

            task = self.task_stack.pop(0)
            tags = [
                t for t in task.tags_to_process if t.id not in self.identified_tags]

            if not tags:
                self.metrics['idle_slots'] += 1
                return self._create_step_result('idle_slot', reader_bits=CONSTANTS.READER_CMD_BASE_BITS)

            common_prefix, _ = RfidUtils.get_collision_info(
                [t.id for t in tags])

            if len(tags) == 1:
                tag = tags[0]
                self.tag_response_counts[tag.id] += 1

                remaining_len = self.id_length - len(common_prefix)
                perfect_response = tag.id[len(common_prefix):]
                noisy_response = apply_ber_noise(perfect_response, self.ber)

                reader_bits = CONSTANTS.READER_CMD_BASE_BITS + \
                    len(common_prefix)

                if perfect_response == noisy_response:
                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] += 1
                    return self._create_step_result('success_slot', reader_bits, remaining_len, remaining_len)
                else:
                    self.task_stack.insert(0, EAQTask(tags_to_process=tags))
                    self.metrics['collision_slots'] += 1
                    return self._create_step_result('collision_slot', reader_bits, remaining_len, remaining_len,
                                                    operation_description="Success slot failed due to BER")

            self.qtcbb_task = EAQTask(tags_to_process=tags)
            self.qtcbb_prefix = common_prefix
            self.current_mode = 'QTCBB_GET_BLOCK_INFO'
            return self.perform_step()

        elif self.current_mode == 'QTCBB_GET_BLOCK_INFO':
            tags = self.qtcbb_task.tags_to_process
            prefix_len = len(self.qtcbb_prefix)
            remaining_len = self.id_length - prefix_len

            for tag in tags:
                self.tag_response_counts[tag.id] += 1

            self.qtcbb_b_i, self.qtcbb_b_l = self._get_collided_block_info(
                [t.id for t in tags], self.qtcbb_prefix)

            self.current_mode = 'QTCBB_GET_BITMAP'
            self.metrics['collision_slots'] += 1

            reader_bits = CONSTANTS.READER_CMD_BASE_BITS + prefix_len
            tag_bits = len(tags) * remaining_len
            return self._create_step_result('collision_slot', reader_bits, tag_bits, remaining_len)

        elif self.current_mode == 'QTCBB_GET_BITMAP':
            tags = self.qtcbb_task.tags_to_process

            if self.qtcbb_b_i == -1:

                self.current_mode = 'PROCESSING_PARTITIONS'
                first_coll_pos, _ = RfidUtils.get_collision_info(
                    [t.id for t in tags])
                split_pos = len(first_coll_pos)
                group_0 = [t for t in tags if t.id[split_pos] == '0']
                group_1 = [t for t in tags if t.id[split_pos] == '1']
                if group_1:
                    self.task_stack.insert(0, EAQTask(tags_to_process=group_1))
                if group_0:
                    self.task_stack.insert(0, EAQTask(tags_to_process=group_0))
                return self._create_step_result('internal_op', operation_description="Fallback to binary split.")

            b_l_to_use = min(self.qtcbb_b_l, self.b_l)
            bitmap_len = 2**b_l_to_use

            aggregated_mrm = [0] * bitmap_len
            for tag in tags:
                self.tag_response_counts[tag.id] += 1
                id_segment = tag.id[self.qtcbb_b_i: self.qtcbb_b_i + b_l_to_use]
                aggregated_mrm[int(id_segment, 2)] = 1

            for i in range(bitmap_len - 1, -1, -1):
                if aggregated_mrm[i] == 1:
                    child_suffix = format(i, f'0{b_l_to_use}b')
                    new_prefix = (self.qtcbb_prefix +
                                  tags[0].id[len(self.qtcbb_prefix):self.qtcbb_b_i] +
                                  child_suffix)
                    child_tags = [
                        t for t in tags if t.id.startswith(new_prefix)]
                    if child_tags:
                        self.task_stack.insert(
                            0, EAQTask(tags_to_process=child_tags))

            reader_bits = CONSTANTS.READER_CMD_BASE_BITS + \
                len(self.qtcbb_prefix) + 20
            tag_bits = len(tags) * bitmap_len

            self.metrics['collision_slots'] += 1
            self.current_mode = 'PROCESSING_PARTITIONS'
            return self._create_step_result('collision_slot', reader_bits, tag_bits, bitmap_len)

        return self._create_step_result('internal_op', operation_description="Error: Unknown state.")
