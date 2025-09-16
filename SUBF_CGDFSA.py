import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


from Framework import TraditionalAlgorithmInterface, AlgorithmStepResult, Tag, CONSTANTS
from Tool import RfidUtils


def get_optimal_frame_size(tag_count: int) -> int:
    """
    根据标签数量，从论文的Table 5中查找最优帧长 (F = 2^Q)。
    """
    if tag_count <= 3:
        return 2
    if tag_count <= 5:
        return 4
    if tag_count <= 11:
        return 8
    if tag_count <= 22:
        return 16
    if tag_count <= 44:
        return 32
    if tag_count <= 89:
        return 64
    if tag_count <= 177:
        return 128
    if tag_count <= 355:
        return 256
    if tag_count <= 710:
        return 512
    if tag_count <= 1420:
        return 1024
    if tag_count <= 2839:
        return 2048
    if tag_count <= 5678:
        return 4096
    if tag_count <= 11357:
        return 8192
    if tag_count <= 22713:
        return 16384
    return 32768


def get_subframe_size(frame_size: int) -> int:
    """
    根据帧长，从论文的Table 4中查找子帧长度。
    """
    if frame_size <= 8:
        return frame_size // 2
    if frame_size <= 16:
        return frame_size // 4
    if frame_size <= 128:
        return frame_size // 8
    if frame_size <= 512:
        return frame_size // 16
    if frame_size <= 1024:
        return frame_size // 32
    return frame_size // 64


def get_group_count_bits(tag_count: int) -> int:
    """
    根据标签总数，从论文的Table 3 (Scheme III) 中确定用于分组的CRC比特数。
    返回0意味着不进行分组。
    """
    if tag_count <= 400:
        return 0
    if tag_count <= 800:
        return 1
    if tag_count <= 1600:
        return 2
    return 3


def calculate_crc(data: str, poly: int, bits: int) -> int:
    """
    一个简化的CRC计算，用于标签分组。
    在真实硬件中，这是一个标准操作。
    """
    crc = 0
    for bit in data:
        crc ^= int(bit)
        if crc & (1 << (bits)):
            crc = (crc << 1) ^ poly
        else:
            crc <<= 1
    return crc & ((1 << bits) - 1)


class SUBF_CGDFSA_Algorithm(TraditionalAlgorithmInterface):
    """
    SUBF-CGDFSA 算法的完整复现。
    该算法基于动态帧时隙ALOHA，并结合了CRC分组和子帧观察机制。
    """

    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)

        self.state = 'INITIAL_ESTIMATION'
        self.groups: Dict[int, List[Tag]] = {}
        self.group_queue: List[int] = []

        self.current_group_id: int = -1
        self.tags_in_current_group: List[Tag] = []
        self.frame_size: int = 0
        self.subframe_size: int = 0
        self.slot_cursor: int = 0

        self.subframe_success_count: int = 0
        self.subframe_collision_count: int = 0

        self.pending_reader_bits: float = 0.0

        self.crc_poly = 19
        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0

    def is_finished(self) -> bool:

        return len(self.identified_tags) == len(self.tags_in_field)

    def perform_step(self) -> AlgorithmStepResult:
        """
        算法的核心状态机，每次调用模拟一个时隙的事件。
        """

        if self.state == 'INITIAL_ESTIMATION':

            total_tags = len(self.tags_in_field)

            group_bits = get_group_count_bits(total_tags)

            if group_bits == 0:

                self.groups[0] = list(self.tags_in_field)
            else:

                for tag in self.tags_in_field:
                    group_id = calculate_crc(tag.id, self.crc_poly, group_bits)
                    if group_id not in self.groups:
                        self.groups[group_id] = []
                    self.groups[group_id].append(tag)

            self.group_queue = sorted(self.groups.keys())
            self.state = 'START_NEW_GROUP'
            return AlgorithmStepResult('internal_op', operation_description=f"完成分组，共 {len(self.groups)} 组")

        elif self.state == 'START_NEW_GROUP':

            if not self.group_queue and self.current_group_id == -1:

                if not self.is_finished():
                    print("警告: 所有组处理完毕，但仍有标签未识别。")
                return AlgorithmStepResult('internal_op', operation_description="所有组处理完毕")

            if not self.tags_in_current_group:
                if not self.group_queue:
                    self.current_group_id = -1
                    return AlgorithmStepResult('internal_op', operation_description="完成最后一组")

                self.current_group_id = self.group_queue.pop(0)

                self.tags_in_current_group = [
                    t for t in self.groups[self.current_group_id] if t.id not in self.identified_tags]

            if not self.tags_in_current_group:
                return AlgorithmStepResult('internal_op', operation_description=f"组 {self.current_group_id} 已空, 跳过")

            num_tags_in_group = len(self.tags_in_current_group)
            self.frame_size = get_optimal_frame_size(num_tags_in_group)
            self.subframe_size = get_subframe_size(self.frame_size)

            self.slot_cursor = 0
            self.subframe_success_count = 0
            self.subframe_collision_count = 0

            self.pending_reader_bits = CONSTANTS.QUERY_CMD_BITS

            self.state = 'RUNNING_FRAME'
            return AlgorithmStepResult('internal_op', operation_description=f"开始识别组 {self.current_group_id}, 帧长={self.frame_size}")

        elif self.state == 'RUNNING_FRAME':
            self.slot_cursor += 1

            responding_tags = []
            for tag in self.tags_in_current_group:
                if random.randint(0, self.frame_size - 1) == 0:
                    responding_tags.append(tag)

            num_responses = len(responding_tags)
            op_type = ''
            tag_bits_cost = 0

            if num_responses == 0:
                op_type = 'idle_slot'
            elif num_responses == 1:
                op_type = 'success_slot'
                tag = responding_tags[0]
                self.identified_tags.add(tag.id)
                self.tags_in_current_group.remove(tag)
                tag_bits_cost = self.id_length
                if self.slot_cursor <= self.subframe_size:
                    self.subframe_success_count += 1
            else:
                op_type = 'collision_slot'
                tag_bits_cost = num_responses * self.id_length
                if self.slot_cursor <= self.subframe_size:
                    self.subframe_collision_count += 1

            reader_bits_cost = self.pending_reader_bits
            self.pending_reader_bits = 0.0

            result = AlgorithmStepResult(
                operation_type=op_type,
                reader_bits=reader_bits_cost,
                tag_bits=tag_bits_cost,
                expected_max_tag_bits=self.id_length,
                operation_description=f"组 {self.current_group_id}, 帧 {self.frame_size}, 时隙 {self.slot_cursor}"
            )

            if self.slot_cursor == self.subframe_size:

                estimated_tags = (self.subframe_success_count + 2.39 *
                                  self.subframe_collision_count) * (self.frame_size / self.subframe_size)
                optimal_f = get_optimal_frame_size(round(estimated_tags))

                if optimal_f != self.frame_size:
                    self.state = 'START_NEW_GROUP'
                    result.operation_description += " (子帧观察: 帧长不优，提前终止)"
                    return result

            if self.slot_cursor >= self.frame_size:
                self.state = 'START_NEW_GROUP'

            return result

        return AlgorithmStepResult('internal_op', operation_description="错误: 未知状态")
