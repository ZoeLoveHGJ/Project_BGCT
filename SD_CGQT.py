import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from Framework import TraditionalAlgorithmInterface, AlgorithmStepResult, Tag, CONSTANTS
from Tool import RfidUtils


@dataclass
class SDCGQTTask:
    prefix: str
    tags_to_process: List[Tag]

class SDCGQTAlgorithm(TraditionalAlgorithmInterface):
    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)
        
        self.K = kwargs.get('K', 4)
        if not (self.K > 0 and (self.K & (self.K - 1)) == 0):
            raise ValueError(f"K必须是2的整数次幂, 但得到的是 {self.K}")

        self.state = 'INITIAL_PROBE'
        
        # 任务堆栈初始为空，在第一次探测后被填充
        self.query_stack: List[SDCGQTTask] = []
        
        self.group_queries: List[str] = []
        self.characteristic_groups: Dict[str, List[str]] = {}
        self.segment_to_group_query: Dict[str, str] = {}
        self._generate_characteristic_groups()

        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0
        self.tag_response_counts: Dict[str, int] = {t.id: 0 for t in tags_in_field}

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

    def _generate_characteristic_groups(self):
        num_vectors = 2**self.K
        all_vectors = [format(i, f'0{self.K}b') for i in range(num_vectors)]
        processed = {vec: False for vec in all_vectors}
        
        for q_candidate in all_vectors:
            if not processed[q_candidate]:
                q = q_candidate
                self.group_queries.append(q)
                group = []
                for i in range(self.K):
                    member_val = int(q, 2) ^ (1 << i)
                    member_vec = all_vectors[member_val]
                    group.append(member_vec)
                    processed[member_vec] = True
                    self.segment_to_group_query[member_vec] = q
                self.characteristic_groups[q] = group
                processed[q] = True

    def perform_step(self) -> AlgorithmStepResult:
        if self.is_finished():
            return AlgorithmStepResult('internal_op', operation_description="所有标签已识别。")
        
        if self.state == 'INITIAL_PROBE':
            # 执行一次全局探测，计算其成本，并为后续步骤填充任务堆栈
            return self._handle_initial_probe()

        if not self.query_stack:
            if len(self.identified_tags) < len(self.tags_in_field):
                print(f"警告: 任务堆栈为空，但仍有 {len(self.tags_in_field) - len(self.identified_tags)} 个标签未识别。")
            return AlgorithmStepResult('internal_op', operation_description="任务堆栈为空。")

        task = self.query_stack.pop(0)
        prefix = task.prefix
        tags = [t for t in task.tags_to_process if t.id not in self.identified_tags]
        num_tags = len(tags)

        if num_tags == 0:
            return AlgorithmStepResult('internal_op', operation_description="空任务弹出。")

        # 每次处理一个任务时，增加相关标签的响应计数
        for tag in tags:
            self.tag_response_counts[tag.id] += 1

        if num_tags == 1:
            tag = tags[0]
            self.identified_tags.add(tag.id)
            self.metrics['success_slots'] += 1
            reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(prefix)
            tag_bits = self.id_length - len(prefix)
            return AlgorithmStepResult('success_slot', reader_bits, tag_bits, tag_bits, f"成功: {tag.id}")

        prefix_len = len(prefix)
        
        if prefix_len + self.K > self.id_length:
            return self._handle_binary_split(tags, prefix)

        next_segments = {t.id[prefix_len : prefix_len + self.K] for t in tags}
        
        collided_bits_in_segment = 0
        if len(next_segments) > 1:
            for i in range(self.K):
                if len({seg[i] for seg in next_segments}) > 1:
                    collided_bits_in_segment += 1
        
        if collided_bits_in_segment == self.K:
            return self._handle_group_query(tags, prefix)
        elif 1 <= collided_bits_in_segment < self.K:
            return self._handle_partial_collision_split(tags, prefix, list(next_segments))
        else:
            new_prefix = prefix + list(next_segments)[0]
            self.query_stack.insert(0, SDCGQTTask(prefix=new_prefix, tags_to_process=tags))
            return AlgorithmStepResult('internal_op', operation_description="段内无碰撞，推进前缀。")

    def _handle_initial_probe(self) -> AlgorithmStepResult:
        # 切换状态，确保此块仅执行一次
        self.state = 'TREE_TRAVERSAL'
        
        # 如果一开始就没有标签，直接结束
        if not self.tags_in_field:
            return AlgorithmStepResult('internal_op', operation_description="场内无标签。")
            
        self.metrics['collision_slots'] += 1

        # --- 成本计算 ---
        # 读写器只发送一个基础的 Query("") 命令
        reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS
        # 所有标签都尝试返回其完整的ID
        actual_tag_bits_cost = len(self.tags_in_field) * self.id_length
        # 时隙的长度由单个标签返回其完整ID的时间决定
        expected_window = self.id_length

        # 增加所有标签的响应计数（这是它们的第一次响应）
        for tag in self.tags_in_field:
            self.tag_response_counts[tag.id] += 1

        # --- 填充初始任务到堆栈 ---
        # 分析这次全局碰撞，找到第一个碰撞点，然后进行二叉分裂
        all_ids = [t.id for t in self.tags_in_field]
        common_prefix, _ = RfidUtils.get_collision_info(all_ids)
        
        # 如果所有ID都一样（虽然不太可能，但作为边界情况处理）
        if len(common_prefix) == self.id_length:
             if self.tags_in_field:
                self.identified_tags.add(self.tags_in_field[0].id)
             # 这种情况只产生一个成功时隙，但这里按碰撞处理，因为有多个标签
             # 实际协议中这依然是碰撞
        else:
            c1 = len(common_prefix) # 第一个碰撞位
            group_0, group_1 = [], []
            for tag in self.tags_in_field:
                if tag.id[c1] == '0':
                    group_0.append(tag)
                else:
                    group_1.append(tag)
            
            new_prefix_0 = common_prefix + '0'
            new_prefix_1 = common_prefix + '1'

            # 将分裂后的任务推入堆栈，注意顺序，通常后进先出
            if group_1: self.query_stack.insert(0, SDCGQTTask(prefix=new_prefix_1, tags_to_process=group_1))
            if group_0: self.query_stack.insert(0, SDCGQTTask(prefix=new_prefix_0, tags_to_process=group_0))

        return AlgorithmStepResult(
            'collision_slot',
            reader_bits=reader_bits_cost,
            tag_bits=actual_tag_bits_cost,
            expected_max_tag_bits=expected_window,
            operation_description="初始探测：所有标签响应，产生全局碰撞"
        )

    def _handle_group_query(self, tags: List[Tag], prefix: str) -> AlgorithmStepResult:
        """处理段内全碰撞 (m=K) 的情况，完整模拟组查询流程。"""
        prefix_len = len(prefix)
        self.metrics['collision_slots'] += 1

        # 模拟物理流程
        num_queries = len(self.group_queries)
        received_sg = [0] * (num_queries * self.K)
        responding_tags_count = 0
        for tag in tags:
            segment = tag.id[prefix_len : prefix_len + self.K]
            if segment in self.segment_to_group_query:
                responding_tags_count += 1
                q = self.segment_to_group_query[segment]
                q_index = self.group_queries.index(q)
                g_i = int(q, 2) ^ int(segment, 2)
                g_i_str = format(g_i, f'0{self.K}b')
                start_pos = q_index * self.K
                for i in range(self.K):
                    if g_i_str[i] == '1':
                        received_sg[start_pos + i] = 1
        
        identified_in_this_step = set()
        if responding_tags_count > 0:
            for i in range(num_queries):
                start_pos = i * self.K
                g_i_block_list = received_sg[start_pos : start_pos + self.K]
                if sum(g_i_block_list) > 0:
                    g_i = int("".join(map(str, g_i_block_list)), 2)
                    q_i = self.group_queries[i]
                    recovered_segment_val = int(q_i, 2) ^ g_i
                    recovered_segment = format(recovered_segment_val, f'0{self.K}b')
                    for tag in tags:
                        if tag.id.startswith(prefix + recovered_segment):
                            self.identified_tags.add(tag.id)
                            identified_in_this_step.add(tag.id)
        
        # 为这个复杂的“宏操作”手动计算真实时间
        reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS + prefix_len
        time_reader_tx = reader_bits_cost / CONSTANTS.READER_BITS_PER_US
        sg_len = 2**self.K
        time_tag_response_window = sg_len / CONSTANTS.TAG_BITS_PER_US
        total_time_us = time_reader_tx + CONSTANTS.T1_US + time_tag_response_window + CONSTANTS.T2_MIN_US
        
        # 返回结果
        if identified_in_this_step:
            remaining_tags = [t for t in tags if t.id not in identified_in_this_step]
            if remaining_tags:
                self.query_stack.insert(0, SDCGQTTask(prefix=prefix, tags_to_process=remaining_tags))
            
            tag_bits_cost = responding_tags_count * sg_len
            
            return AlgorithmStepResult(
                'collision_slot',
                reader_bits=reader_bits_cost,
                tag_bits=tag_bits_cost,
                override_time_us=total_time_us,
                operation_description=f"组查询识别了 {len(identified_in_this_step)} 个标签"
            )
        else:
            # 失败: 组查询未能识别任何标签，强制二叉分裂回退
            self._handle_binary_split_logic(tags, prefix)
            return AlgorithmStepResult(
                'collision_slot',
                reader_bits=reader_bits_cost,
                tag_bits=0,
                override_time_us=total_time_us,
                operation_description="组查询失败，强制二叉分裂回退"
            )

    def _handle_partial_collision_split(self, tags: List[Tag], prefix: str, segments: List[str]) -> AlgorithmStepResult:
        self.metrics['collision_slots'] += 1
        reader_bits_cost = CONSTANTS.READER_CMD_BASE_BITS + len(prefix)
        tag_bits_cost = len(tags) * (self.id_length - len(prefix))
        expected_window = self.id_length - len(prefix)
        
        common_segment_prefix, _ = RfidUtils.get_collision_info(segments)
        c1 = len(common_segment_prefix)
        
        group_0, group_1 = [], []
        for tag in tags:
            if tag.id[len(prefix) + c1] == '0': group_0.append(tag)
            else: group_1.append(tag)
        
        new_prefix_0 = prefix + common_segment_prefix + '0'
        new_prefix_1 = prefix + common_segment_prefix + '1'

        if group_1: self.query_stack.insert(0, SDCGQTTask(prefix=new_prefix_1, tags_to_process=group_1))
        if group_0: self.query_stack.insert(0, SDCGQTTask(prefix=new_prefix_0, tags_to_process=group_0))
        
        return AlgorithmStepResult('collision_slot', reader_bits=reader_bits_cost, tag_bits=tag_bits_cost, expected_max_tag_bits=expected_window)

    def _handle_binary_split_logic(self, tags: List[Tag], prefix: str):
        common_prefix, _ = RfidUtils.get_collision_info([t.id for t in tags])
        
        if len(common_prefix) <= len(prefix) and len(tags) > 1:
            c1 = len(prefix)
        else:
            c1 = len(common_prefix)
        
        if c1 >= self.id_length:
            if tags: self.identified_tags.add(tags[0].id)
            return

        group_0, group_1 = [], []
        for tag in tags:
            if tag.id[c1] == '0': group_0.append(tag)
            else: group_1.append(tag)
        
        new_prefix_0 = common_prefix + '0'
        new_prefix_1 = common_prefix + '1'

        if group_1: self.query_stack.insert(0, SDCGQTTask(prefix=new_prefix_1, tags_to_process=group_1))
        if group_0: self.query_stack.insert(0, SDCGQTTask(prefix=new_prefix_0, tags_to_process=group_0))

    def _handle_binary_split(self, tags: List[Tag], prefix: str) -> AlgorithmStepResult:
        self.metrics['collision_slots'] += 1
        
        # 首先执行分裂逻辑，填充堆栈
        self._handle_binary_split_logic(tags, prefix)

        # 然后计算这一步的成本
        # 注意：这里的成本是基于前缀的查询，而不是全局查询
        reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(prefix)
        expected_tag_bits = self.id_length - len(prefix)
        actual_tag_bits = len(tags) * expected_tag_bits
        
        return AlgorithmStepResult('collision_slot', reader_bits, actual_tag_bits, expected_tag_bits)

