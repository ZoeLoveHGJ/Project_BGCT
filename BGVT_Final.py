

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set


from Framework import TraditionalAlgorithmInterface, AlgorithmStepResult, Tag, CONSTANTS
from Tool import RfidUtils


class BGVT_Algorithm(TraditionalAlgorithmInterface):

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)
        self.d_max = kwargs.get('d_max', 8)
        initial_state = BGVTState(tags_to_identify=self.tags_in_field)
        self.query_stack: List[BGVTState] = [initial_state]
        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0
        self.tag_response_counts: Dict[str, int] = {
            t.id: 0 for t in tags_in_field}
        self.current_mode = 'PLANNING_PHASE'
        self.tags_for_planning: List[Tag] = []
        self.planned_groups: List[List[Tag]] = []
        self.non_idle_groups: List[List[Tag]] = []
        self.sub_slot_cursor: int = 0
        self.current_split_prefix_len: int = 0
        self.current_d_to_use: int = 0
        self.pending_cmd_reader_bits: float = 0.0

        if not hasattr(self, 'metrics'):
            self.metrics = {}
        self.metrics.update({
            'avg_tag_responses': 0,
            'min_tag_responses': 0,
            'max_tag_responses': 0
        })
        self.final_metrics_calculated = False

    def is_finished(self) -> bool:
        """
        检查仿真是否完成，并在完成时计算最终的统计指标。
        """
        if not hasattr(self, 'identified_tags'):
            self.identified_tags = set()

        finished = len(self.identified_tags) == len(self.tags_in_field)

        if finished and not self.final_metrics_calculated:
            self.final_metrics_calculated = True
            counts = list(self.tag_response_counts.values())
            if counts:
                self.metrics['avg_tag_responses'] = np.mean(counts)
                self.metrics['min_tag_responses'] = np.min(counts)
                self.metrics['max_tag_responses'] = np.max(counts)

        return finished


@dataclass
class BGVTState:
    """用于在主堆栈中存储待处理的标签集合。"""
    tags_to_identify: List[Tag] = field(default_factory=list)


def simple_chunk_hash(chunk: str, hash_len: int) -> int:
    """
    计算一个比特串块的简单哈希值。
    在真实场景中，这可能是一个更健壮的CRC算法。
    Args:
        chunk (str): 用于计算哈希的比特串。
        hash_len (int): 哈希值的目标长度（比特）。
    Returns:
        int: 计算出的哈希值。
    """
    if not chunk:
        return 0

    val = int(chunk, 2)
    val = (val ^ (val >> (len(chunk) // 2)))
    val = (val ^ (val >> (len(chunk) // 4)))
    mask = (1 << hash_len) - 1
    return val & mask


class BGVT_Final_Algorithm(BGVT_Algorithm):

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field, **kwargs)
        self.hash_len = kwargs.get('hash_len', 4)
        self.hash_probe_chunk_size = kwargs.get('hash_probe_chunk_size', 16)
        self.planned_hash_groups: Dict[int, List[Tag]] = {}

    def _perform_binary_split_on_group(self, tags: List[Tag], prefix_len: int):
        if not tags:
            return

        tag_ids = [t.id for t in tags]
        common_prefix, _ = RfidUtils.get_collision_info(tag_ids)

        split_pos = len(common_prefix)

        if split_pos >= self.id_length:

            for tag in tags:
                self.identified_tags.add(tag.id)
            return

        if split_pos <= prefix_len:
            split_pos = prefix_len

        group_0, group_1 = [], []
        for tag in tags:
            if tag.id[split_pos] == '0':
                group_0.append(tag)
            else:
                group_1.append(tag)

        if group_1:
            self.query_stack.insert(0, BGVTState(tags_to_identify=group_1))
        if group_0:
            self.query_stack.insert(0, BGVTState(tags_to_identify=group_0))

    def perform_step(self) -> AlgorithmStepResult:
        if self.current_mode == 'PLANNING_PHASE':
            if not self.query_stack:
                return AlgorithmStepResult('internal_op', operation_description="完成")

            current_state = self.query_stack.pop(0)
            tags = [
                t for t in current_state.tags_to_identify if t.id not in self.identified_tags]

            if len(tags) <= 1:
                if len(tags) == 1:
                    tag = tags[0]
                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] = self.metrics.get(
                        'success_slots', 0) + 1
                    self.tag_response_counts[tag.id] = self.tag_response_counts.get(
                        tag.id, 0) + 1
                    reader_bits = CONSTANTS.READER_CMD_BASE_BITS
                    tag_bits = self.id_length
                    return AlgorithmStepResult('success_slot', reader_bits, tag_bits, tag_bits, f"成功识别: {tag.id}")
                return AlgorithmStepResult('internal_op', operation_description="空状态弹出")

            self.tags_for_planning = tags
            self.current_mode = 'HASH_PLANNING_PHASE'
            return AlgorithmStepResult('internal_op', operation_description="启动哈希规划")

        elif self.current_mode == 'HASH_PLANNING_PHASE':
            tags = self.tags_for_planning
            tag_ids = [t.id for t in tags]
            common_prefix, _ = RfidUtils.get_collision_info(tag_ids)
            prefix_len = len(common_prefix)

            start = prefix_len
            end = min(start + self.hash_probe_chunk_size, self.id_length)

            self.planned_hash_groups = {}
            for tag in tags:
                chunk = tag.id[start:end]
                h = simple_chunk_hash(chunk, self.hash_len)
                if h not in self.planned_hash_groups:
                    self.planned_hash_groups[h] = []
                self.planned_hash_groups[h].append(tag)

            if len(self.planned_hash_groups) <= 1:
                self.metrics['collision_slots'] = self.metrics.get(
                    'collision_slots', 0) + 1
                for tag in tags:
                    self.tag_response_counts[tag.id] = self.tag_response_counts.get(
                        tag.id, 0) + 1
                reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS + prefix_len
                tag_bits_cost = len(tags) * self.hash_len
                expected_window = self.hash_len
                self._perform_binary_split_on_group(tags, prefix_len)
                self.current_mode = 'PLANNING_PHASE'
                return AlgorithmStepResult('collision_slot', reader_bits=reader_bits_cost, tag_bits=tag_bits_cost,
                                           expected_max_tag_bits=expected_window,
                                           operation_description=f"哈希探测失败，回退到二叉分裂")

            self.metrics['collision_slots'] = self.metrics.get(
                'collision_slots', 0) + 1
            for tag in tags:
                self.tag_response_counts[tag.id] = self.tag_response_counts.get(
                    tag.id, 0) + 1

            reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS + prefix_len
            tag_bits_cost = len(tags) * self.hash_len
            expected_window = self.hash_len

            self.current_split_prefix_len = prefix_len
            bits_for_hash_params = 16
            self.pending_cmd_reader_bits = CONSTANTS.READER_CMD_BASE_BITS + \
                prefix_len + bits_for_hash_params
            self.current_mode = 'PREDICTION_PHASE'

            return AlgorithmStepResult('collision_slot', reader_bits=reader_bits_cost, tag_bits=tag_bits_cost,
                                       expected_max_tag_bits=expected_window,
                                       operation_description=f"哈希探测成功 (len={self.hash_len})")

        elif self.current_mode == 'PREDICTION_PHASE':
            self.non_idle_groups = list(self.planned_hash_groups.values())

            if not self.non_idle_groups:
                self.current_mode = 'PLANNING_PHASE'
                return AlgorithmStepResult('internal_op', operation_description="预测结果：所有哈希时隙均为空")

            self.current_mode = 'EXECUTING_PHASE'
            self.sub_slot_cursor = 0

            bitmap_len = 2**self.hash_len
            reader_bits = self.pending_cmd_reader_bits
            self.pending_cmd_reader_bits = 0.0

            return AlgorithmStepResult('collision_slot', reader_bits=reader_bits, tag_bits=bitmap_len,
                                       expected_max_tag_bits=bitmap_len,
                                       operation_description=f"接收哈希预测位图 (长度={bitmap_len})")

        elif self.current_mode == 'EXECUTING_PHASE':
            group = self.non_idle_groups[self.sub_slot_cursor]
            self.sub_slot_cursor += 1

            expected_bits = self.id_length - self.current_split_prefix_len
            for tag in group:
                self.tag_response_counts[tag.id] = self.tag_response_counts.get(
                    tag.id, 0) + 1

            if len(group) == 1:
                self.identified_tags.add(group[0].id)
                self.metrics['success_slots'] = self.metrics.get(
                    'success_slots', 0) + 1
                result = AlgorithmStepResult(
                    'success_slot', tag_bits=expected_bits, expected_max_tag_bits=expected_bits)
            else:
                self._perform_binary_split_on_group(
                    group, self.current_split_prefix_len)
                self.metrics['collision_slots'] = self.metrics.get(
                    'collision_slots', 0) + 1
                result = AlgorithmStepResult('collision_slot', tag_bits=len(group) * expected_bits, expected_max_tag_bits=expected_bits,
                                             operation_description="执行阶段碰撞，回退到二叉分裂")

            if self.sub_slot_cursor >= len(self.non_idle_groups):
                self.current_mode = 'PLANNING_PHASE'

            return result

        return AlgorithmStepResult('internal_op', operation_description="错误: 未知状态")
