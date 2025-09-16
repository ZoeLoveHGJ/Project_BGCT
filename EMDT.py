

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
class EMDTTask:
    """用于在查询堆栈中存储任务的数据结构"""
    psc: int
    q: int
    f: int
    tags_to_process: List[Tag]


class EMDTAlgorithm(TraditionalAlgorithmInterface):
    """EMDT 算法的复现实现 - V2 适配版"""

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)

        self.M = kwargs.get('M', 16)
        self.n_threshold = 3

        initial_task = EMDTTask(
            psc=0, q=0, f=1, tags_to_process=self.tags_in_field)
        self.task_stack: List[EMDTTask] = [initial_task]

        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0
        self.tag_response_counts: Dict[str, int] = {
            t.id: 0 for t in tags_in_field}

        self.is_in_collecting_frame = False
        self.collect_frame_slots: List[List[Tag]] = []
        self.collect_frame_cursor = 0
        self.collect_frame_reader_bits = 0.0

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

    def _estimate_tags_ze(self, c0: int) -> float:
        """使用 Zero-Estimation (ZE) 估算标签数。"""
        if c0 == self.M:
            return 0.0
        if c0 == 0:
            return float('inf')
        return math.ceil(math.log(c0 / self.M) / math.log(1.0 - 1.0 / self.M))

    def _perform_scanning_slot(self, task: EMDTTask) -> AlgorithmStepResult:
        """处理一个扫描时隙。"""
        tags = task.tags_to_process
        for tag in tags:
            self.tag_response_counts[tag.id] += 1

        aggregated_ds = [0] * self.M
        tag_responses = {tag.id: random.randint(0, self.M - 1) for tag in tags}
        child_groups = [[] for _ in range(self.M)]
        for tag in tags:
            r = tag_responses[tag.id]
            aggregated_ds[r] = 1
            child_groups[r].append(tag)

        c0 = aggregated_ds.count(0)
        n_hat = self._estimate_tags_ze(c0)
        switch_to_collect = (n_hat / self.M) <= self.n_threshold

        for i in range(self.M - 1, -1, -1):
            if aggregated_ds[i] == 1:
                new_psc = task.psc + i
                child_tags = child_groups[i]
                if switch_to_collect:
                    frame_size = max(1, math.ceil(n_hat / self.M)
                                     if n_hat != float('inf') else len(child_tags))
                    new_task = EMDTTask(
                        psc=new_psc, q=1, f=frame_size, tags_to_process=child_tags)
                else:
                    new_task = EMDTTask(
                        psc=new_psc, q=0, f=1, tags_to_process=child_tags)
                self.task_stack.insert(0, new_task)

        reader_bits = CONSTANTS.READER_CMD_BASE_BITS
        tag_bits = len(tags) * self.M
        expected_max_tag_bits = self.M

        self.metrics['collision_slots'] += 1
        return self._create_step_result('collision_slot', reader_bits, tag_bits, expected_max_tag_bits)

    def _setup_collecting_frame(self, task: EMDTTask):
        """设置一个收集帧。"""
        self.is_in_collecting_frame = True
        self.collect_frame_cursor = 0
        self.collect_frame_slots = [[] for _ in range(task.f)]

        for tag in task.tags_to_process:
            self.tag_response_counts[tag.id] += 1
            self.collect_frame_slots[random.randint(0, task.f - 1)].append(tag)

        self.collect_frame_reader_bits = CONSTANTS.READER_CMD_BASE_BITS

    def _perform_collecting_sub_slot(self) -> AlgorithmStepResult:
        """处理收集帧中的一个子时隙。"""
        tags_in_slot = self.collect_frame_slots[self.collect_frame_cursor]
        num_tags = len(tags_in_slot)

        reader_bits = self.collect_frame_reader_bits
        self.collect_frame_reader_bits = 0

        tag_bits_per_tag = self.id_length

        result = None
        if num_tags == 0:
            self.metrics['idle_slots'] += 1
            result = self._create_step_result('idle_slot', reader_bits)

        elif num_tags == 1:
            tag = tags_in_slot[0]

            perfect_response = tag.id
            noisy_response = apply_ber_noise(perfect_response, self.ber)

            if perfect_response == noisy_response:

                self.metrics['success_slots'] += 1
                self.identified_tags.add(tag.id)
                result = self._create_step_result(
                    'success_slot', reader_bits, tag_bits_per_tag, tag_bits_per_tag)
            else:

                self.metrics['collision_slots'] += 1
                new_task = EMDTTask(
                    psc=0, q=1, f=2, tags_to_process=tags_in_slot)
                self.task_stack.insert(0, new_task)
                result = self._create_step_result('collision_slot', reader_bits, tag_bits_per_tag, tag_bits_per_tag,
                                                  operation_description="Success slot failed due to BER")
        else:
            self.metrics['collision_slots'] += 1
            new_task = EMDTTask(psc=0, q=1, f=2, tags_to_process=tags_in_slot)
            self.task_stack.insert(0, new_task)
            result = self._create_step_result(
                'collision_slot', reader_bits, 0, tag_bits_per_tag)

        self.collect_frame_cursor += 1
        if self.collect_frame_cursor >= len(self.collect_frame_slots):
            self.is_in_collecting_frame = False

        return result

    def perform_step(self) -> AlgorithmStepResult:
        if self.is_finished():
            return self._create_step_result('internal_op', operation_description="All tags identified.")

        if self.is_in_collecting_frame:
            return self._perform_collecting_sub_slot()

        if not self.task_stack:
            return self._create_step_result('internal_op', operation_description="Task stack empty.")

        current_task = self.task_stack.pop(0)
        tags_to_process = [
            t for t in current_task.tags_to_process if t.id not in self.identified_tags]

        if not tags_to_process:
            return self._create_step_result('internal_op', operation_description="No tags to process in this task.")

        current_task.tags_to_process = tags_to_process

        if len(tags_to_process) == 1:
            tag = tags_to_process[0]
            self.identified_tags.add(tag.id)
            self.metrics['success_slots'] += 1
            self.tag_response_counts[tag.id] += 1
            return self._create_step_result('success_slot', CONSTANTS.READER_CMD_BASE_BITS, self.id_length, self.id_length)

        if current_task.q == 0:
            return self._perform_scanning_slot(current_task)
        else:
            self._setup_collecting_frame(current_task)
            return self._perform_collecting_sub_slot()
