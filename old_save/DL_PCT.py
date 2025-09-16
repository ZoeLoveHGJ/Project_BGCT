# -*- coding: utf-8 -*-
"""
DL-PCT (Deep-Locking Parallel Collision Tree) 算法的实现。

*** V2.2 - 框架兼容版 ***

该版本核心升级:
1.  严格遵循用户提供的框架V2.0进行重构，解决了与 calculate_time_delta 函数的逻辑冲突。
2.  QueryDL 指令的开销被正确地附加到第一个子时隙上，确保了时间和能耗统计的精确性。
3.  引入 self.pending_querydl_reader_bits 变量来暂存指令开销。
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set

# 假设这些类从您的框架文件中被正确导入
from Framework import TraditionalAlgorithmInterface, AlgorithmStepResult, Tag, CONSTANTS
from Tool import RfidUtils

@dataclass
class DLPCTState:
    """
    用于在主堆栈中存储待处理的标签集合。
    """
    tags_to_identify: List[Tag] = field(default_factory=list)

class DL_PCTAlgorithm(TraditionalAlgorithmInterface):
    """
    DL-PCT 算法的重构实现，包含一个内部状态机以精确模拟协议流程。
    """
    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)
        self.d_max = kwargs.get('d_max', 8)
        initial_state = DLPCTState(tags_to_identify=self.tags_in_field)
        self.query_stack: List[DLPCTState] = [initial_state]
        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0
        self.tag_response_counts: Dict[str, int] = {t.id: 0 for t in tags_in_field}

        # --- 状态机核心变量 ---
        self.current_mode = 'PLANNING_PHASE'
        self.planned_groups: List[List[Tag]] = []
        self.sub_slot_cursor: int = 0
        self.current_split_prefix_len: int = 0
        
        # --- 新增变量，用于暂存QueryDL指令的比特开销 ---
        self.pending_querydl_reader_bits: float = 0.0

    def is_finished(self) -> bool:
        finished = len(self.identified_tags) == len(self.tags_in_field)
        if finished and 'avg_tag_responses' not in self.metrics:
            counts = list(self.tag_response_counts.values())
            if counts:
                self.metrics['avg_tag_responses'] = np.mean(counts)
                self.metrics['min_tag_responses'] = np.min(counts)
                self.metrics['max_tag_responses'] = np.max(counts)
            else:
                self.metrics.update({'avg_tag_responses': 0, 'min_tag_responses': 0, 'max_tag_responses': 0})
        return finished

    def perform_step(self) -> AlgorithmStepResult:
        # ======================================================================
        # 状态一: 正在依次执行一个已规划好的并行分裂任务
        # ======================================================================
        if self.current_mode == 'EXECUTING_PHASE':
            current_group = self.planned_groups[self.sub_slot_cursor]
            num_tags_in_group = len(current_group)
            self.sub_slot_cursor += 1

            expected_max_tag_bits = self.id_length - self.current_split_prefix_len
            
            # 从暂存变量中取出QueryDL的开销，并立即清零
            # 这个开销只会被加到第一个子时隙上
            reader_bits_cost = self.pending_querydl_reader_bits
            self.pending_querydl_reader_bits = 0.0
            
            result = None

            # Case 1: 空闲子时隙
            if num_tags_in_group == 0:
                self.metrics['idle_slots'] += 1
                result = AlgorithmStepResult('idle_slot', 
                                             reader_bits=reader_bits_cost, 
                                             expected_max_tag_bits=expected_max_tag_bits)
            # Case 2: 成功子时隙
            elif num_tags_in_group == 1:
                tag = current_group[0]
                self.identified_tags.add(tag.id)
                self.metrics['success_slots'] += 1
                actual_tag_bits = self.id_length - self.current_split_prefix_len
                result = AlgorithmStepResult('success_slot',
                                             reader_bits=reader_bits_cost,
                                             tag_bits=actual_tag_bits,
                                             expected_max_tag_bits=expected_max_tag_bits)
            # Case 3: 残余碰撞子时隙
            else:
                self.query_stack.insert(0, DLPCTState(tags_to_identify=current_group))
                self.metrics['collision_slots'] += 1
                total_collision_bits = num_tags_in_group * (self.id_length - self.current_split_prefix_len)
                result = AlgorithmStepResult('collision_slot',
                                             reader_bits=reader_bits_cost,
                                             tag_bits=total_collision_bits,
                                             expected_max_tag_bits=expected_max_tag_bits)
            
            # 如果这是最后一个子时隙，处理完后将模式切换回规划阶段
            if self.sub_slot_cursor == len(self.planned_groups):
                self.current_mode = 'PLANNING_PHASE'
                self.planned_groups = []
            
            return result

        # ======================================================================
        # 状态二: 处于规划阶段，准备处理一个新的碰撞
        # ======================================================================
        elif self.current_mode == 'PLANNING_PHASE':
            if not self.query_stack:
                return AlgorithmStepResult('internal_op', operation_description="Finished")

            current_state = self.query_stack.pop(0)
            tags_in_this_state = [t for t in current_state.tags_to_identify if t.id not in self.identified_tags]
            num_tags = len(tags_in_this_state)

            if num_tags == 0:
                return AlgorithmStepResult('internal_op', operation_description="Empty state popped")

            # 简单情况: 只有一个标签，直接识别
            if num_tags == 1:
                tag = tags_in_this_state[0]
                self.identified_tags.add(tag.id)
                self.metrics['success_slots'] += 1
                self.tag_response_counts[tag.id] += 1
                reader_bits = CONSTANTS.READER_CMD_BASE_BITS
                actual_tag_bits = self.id_length
                return AlgorithmStepResult('success_slot', reader_bits, actual_tag_bits, actual_tag_bits)

            # --- 复杂情况：遇到碰撞，进入“规划”阶段 ---
            tag_ids = [t.id for t in tags_in_this_state]
            common_prefix, collision_positions = RfidUtils.get_collision_info(tag_ids)
            d_detected = len(collision_positions)
            d_to_use = min(d_detected, self.d_max)
            if d_to_use == 0:
                collision_positions = [len(common_prefix)]
                d_to_use = 1
            positions_to_use = collision_positions[:d_to_use]

            num_sub_slots = 2**d_to_use
            groups = [[] for _ in range(num_sub_slots)]
            for tag in tags_in_this_state:
                feature_vector = "".join([tag.id[i] for i in positions_to_use])
                group_index = int(feature_vector, 2)
                groups[group_index].append(tag)

            # 保存规划结果
            self.planned_groups = groups
            self.sub_slot_cursor = 0
            self.current_split_prefix_len = len(common_prefix)
            
            # 计算并暂存QueryDL指令的开销
            bits_for_d = 5
            bits_per_pos = 7
            self.pending_querydl_reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(common_prefix) + bits_for_d + d_to_use * bits_per_pos

            # 切换模式，以便下一次调用时开始执行子时隙
            self.current_mode = 'EXECUTING_PHASE'

            for tag in tags_in_this_state:
                self.tag_response_counts[tag.id] += 1

            # 返回一个零开销的内部操作，仅用于推进仿真循环
            # 真正的开销将在下一个 EXECUTING_PHASE 步骤中被计算
            return AlgorithmStepResult('internal_op', 
                                       operation_description=f"Planned QueryDL, d={d_to_use}")
        
        # 兜底，理论上不应到达这里
        return AlgorithmStepResult('internal_op', operation_description="Error: Unknown state")
