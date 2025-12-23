import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


from Framework import (
    TraditionalAlgorithmInterface,
    AlgorithmStepResult,
    Tag,
    CONSTANTS
)


@dataclass
class EMDTTask:
    psc: int
    q: int
    f: int
    tags_to_process: List[Tag]


class EMDTAlgorithm(TraditionalAlgorithmInterface):

    def __init__(self, tags_in_field: List[Tag], **kwargs):

        super().__init__(tags_in_field, **kwargs)

        self.M = kwargs.get('M', 16)
        self.n_threshold = 3

        initial_task = EMDTTask(
            psc=0, q=0, f=1, tags_to_process=self.tags_in_field)
        self.task_stack: List[EMDTTask] = [initial_task]

        self.id_length = len(tags_in_field[0].id) if tags_in_field else 96
        self.tag_response_counts: Dict[str, int] = {
            t.id: 0 for t in tags_in_field}

        self.metrics['total_tag_responses'] = 0

        self.is_in_collecting_frame = False
        self.collect_frame_slots: List[List[Tag]] = []
        self.collect_frame_cursor = 0
        self.collect_frame_reader_bits = 0.0

        self.enable_monitoring = kwargs.get(
            'enable_resource_monitoring', False)

    def _create_step_result(self, *args, **kwargs) -> AlgorithmStepResult:
        """辅助函数：创建结果对象并附加资源监控指标。"""
        result = AlgorithmStepResult(*args, **kwargs)
        if self.enable_monitoring:
            result.internal_metrics = {'stack_depth': len(self.task_stack)}
        return result

    def is_finished(self) -> bool:
        """检查是否所有标签都已被识别。"""
        finished = len(self.identified_tags) == len(self.tags_in_field)

        if finished and 'avg_tag_responses' not in self.metrics:
            if self.total_tag_count > 0:
                self.metrics['avg_tag_responses'] = self.metrics['total_tag_responses'] / \
                    self.total_tag_count
            else:
                self.metrics['avg_tag_responses'] = 0
        return finished

    @property
    def total_tag_count(self):
        return len(self.tags_in_field)

    def _estimate_tags_ze(self, c0: int) -> float:
        """Zero-Estimation (ZE) 估算算法。"""
        if c0 == self.M:
            return 0.0
        if c0 == 0:
            return float('inf')
        return math.ceil(math.log(c0 / self.M) / math.log(1.0 - 1.0 / self.M))

    def _perform_scanning_slot(self, task: EMDTTask) -> AlgorithmStepResult:

        active_tags = self.channel.filter_active_tags(task.tags_to_process)

        if active_tags:
            self.metrics['total_tag_responses'] += len(active_tags)
            for tag in active_tags:
                self.tag_response_counts[tag.id] += 1

        tag_selections = {}
        virtual_ds_tags = []

        child_groups = [[] for _ in range(self.M)]

        for tag in active_tags:
            r = random.randint(0, self.M - 1)
            tag_selections[tag.id] = r

            ds_bits = ['0'] * self.M
            ds_bits[r] = '1'
            virtual_ds_string = "".join(ds_bits)

            virtual_ds_tags.append(Tag(virtual_ds_string))
            child_groups[r].append(tag)

        if not virtual_ds_tags:
            observed_ds = "0" * self.M
        else:
            observed_ds, _ = self.channel.resolve_collision(virtual_ds_tags)

        c0 = observed_ds.count('0')
        n_hat = self._estimate_tags_ze(c0)

        switch_to_collect = (n_hat / self.M) <= self.n_threshold

        for i in range(self.M - 1, -1, -1):
            char_at_pos = observed_ds[i]

            if char_at_pos != '0':
                new_psc = task.psc + i
                child_tags = child_groups[i] if i < len(child_groups) else []

                if switch_to_collect:
                    if n_hat != float('inf'):
                        frame_size = max(1, math.ceil(n_hat / self.M))
                    else:
                        frame_size = max(1, len(child_tags))

                    new_task = EMDTTask(
                        psc=new_psc, q=1, f=frame_size, tags_to_process=child_tags)
                else:
                    new_task = EMDTTask(
                        psc=new_psc, q=0, f=1, tags_to_process=child_tags)

                self.task_stack.insert(0, new_task)

        reader_bits = CONSTANTS.READER_CMD_BASE_BITS
        tag_bits = len(active_tags) * self.M

        self.metrics['collision_slots'] += 1

        return self._create_step_result(
            'collision_slot',
            reader_bits,
            tag_bits,
            self.M,
            operation_description=f"Scanning M={self.M}, Est={n_hat:.1f}"
        )

    def _setup_collecting_frame(self, task: EMDTTask):
        self.is_in_collecting_frame = True
        self.collect_frame_cursor = 0
        self.collect_frame_slots = [[] for _ in range(task.f)]

        for tag in task.tags_to_process:
            if task.f > 0:
                slot_idx = random.randint(0, task.f - 1)
                self.collect_frame_slots[slot_idx].append(tag)

        self.collect_frame_reader_bits = CONSTANTS.READER_CMD_BASE_BITS

    def _perform_collecting_sub_slot(self) -> AlgorithmStepResult:

        candidate_tags = self.collect_frame_slots[self.collect_frame_cursor]

        active_tags = self.channel.filter_active_tags(candidate_tags)

        if active_tags:
            self.metrics['total_tag_responses'] += len(active_tags)
            for tag in active_tags:
                self.tag_response_counts[tag.id] += 1

        reader_bits = self.collect_frame_reader_bits
        self.collect_frame_reader_bits = 0
        tag_bits_per_tag = self.id_length

        result = None

        observed_signal, collision_indices = self.channel.resolve_collision(
            active_tags)

        if not active_tags:
            self.metrics['idle_slots'] += 1
            result = self._create_step_result('idle_slot', reader_bits)

        elif 'X' in observed_signal:
            self.metrics['collision_slots'] += 1

            new_task = EMDTTask(
                psc=0, q=1, f=2, tags_to_process=candidate_tags)
            self.task_stack.insert(0, new_task)

            result = self._create_step_result(
                'collision_slot',
                reader_bits,
                0,
                tag_bits_per_tag,
                operation_description="Physical Collision Detected"
            )

        else:
            decoded_id_matches = any(
                t.id == observed_signal for t in active_tags)

            if decoded_id_matches:

                self.metrics['success_slots'] += 1
                self.identified_tags.add(observed_signal)

                remaining_tags = [
                    t for t in candidate_tags if t.id != observed_signal]
                if remaining_tags:
                    new_task = EMDTTask(
                        psc=0, q=1, f=2, tags_to_process=remaining_tags)
                    self.task_stack.insert(0, new_task)

                result = self._create_step_result(
                    'success_slot',
                    reader_bits,
                    tag_bits_per_tag,
                    tag_bits_per_tag,
                    operation_description="Tag Identified"
                )
            else:

                self.metrics['collision_slots'] += 1
                new_task = EMDTTask(
                    psc=0, q=1, f=2, tags_to_process=candidate_tags)
                self.task_stack.insert(0, new_task)

                result = self._create_step_result(
                    'collision_slot',
                    reader_bits,
                    tag_bits_per_tag,
                    tag_bits_per_tag,
                    operation_description="Corrupted ID (CRC Fail)"
                )

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
            return self._create_step_result('internal_op', operation_description="Empty task skipped.")

        current_task.tags_to_process = tags_to_process

        if current_task.q == 0:
            return self._perform_scanning_slot(current_task)
        else:
            self._setup_collecting_frame(current_task)
            return self._perform_collecting_sub_slot()
