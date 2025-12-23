
"""
RFID仿真框架核心
引入了“信道中间件”层，以支持“真实物理层特性”仿真。
1.  **信道抽象层 (Channel Middleware)**:
    - 引入 `BaseChannel`, `IdealChannel`, `NoisyChannel` 类。
    - 将物理层非理想因素（不响应、误码、碰撞漏检）完全封装在信道类内部。
    - 算法通过 `self.channel` 与标签交互，实现了算法逻辑与环境噪声的解耦。

2.  **物理层损伤模型**:
    - **Tag Dropout**: 模拟标签能量不足或信道衰落导致的不响应。
    - **Collision Missed Detection**: 模拟阅读器将碰撞位(X)误判为0或1。
    - **Guard Interval**: 模拟时序抖动(Jitter)带来的额外时间开销。

3.  **兼容性**:
    - 保留了 'dispersed' 等所有 ID 分布模式生成逻辑。
    - 接口保持向后兼容，未配置噪声参数时自动回退到理想模式。
"""

import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple


@dataclass(frozen=True)
class SimulationConstants:
    """严格按照EPC C1G2标准定义的仿真物理层和时序常量。"""
    TARI_US: float = 12.5

    @property
    def RTCAL_US(self) -> float:
        return 2.5 * self.TARI_US

    @property
    def READER_TO_TAG_BPS(self) -> float:
        return 1.0 / (self.TARI_US * 1e-6)

    @property
    def TAG_TO_READER_BPS(self) -> float:
        return 2.0 * self.READER_TO_TAG_BPS

    @property
    def T1_US(self) -> float:
        return max(self.RTCAL_US, 10.0)

    @property
    def T2_MIN_US(self) -> float:
        return self.T1_US + 20.0

    QUERY_CMD_BITS: int = 22
    QUERYREP_CMD_BITS: int = 4
    ACK_CMD_BITS: int = 18
    RN16_RESPONSE_BITS: int = 16
    EPC_CODE_BITS: int = 96

    @property
    def READER_CMD_BASE_BITS(self) -> int:
        return self.QUERY_CMD_BITS

    @property
    def READER_BITS_PER_US(self) -> float:
        return self.READER_TO_TAG_BPS / 1.0e6

    @property
    def TAG_BITS_PER_US(self) -> float:
        return self.TAG_TO_READER_BPS / 1.0e6


CONSTANTS = SimulationConstants()


class Tag:
    """代表一个RFID标签的简单类。"""

    def __init__(self, identity_code: str):
        self.id: str = identity_code


@dataclass
class AlgorithmStepResult:
    """封装了算法单步执行后产生的结果。"""
    operation_type: str
    reader_bits: float = 0.0
    tag_bits: float = 0.0
    expected_max_tag_bits: int = 0
    operation_description: str = ''
    override_time_us: Optional[float] = None
    internal_metrics: Optional[Dict] = field(default_factory=dict)


class BaseChannel:
    """
    信道基类：定义算法与物理环境交互的标准接口。
    """

    def __init__(self):

        self.ber = 0.0
        self.miss_rate = 0.0
        self.dropout_rate = 0.0

    def filter_active_tags(self, tags: List[Tag]) -> List[Tag]:
        """决定哪些标签在当前时隙实际上能响应（处理 Dropout）。"""
        return tags

    def resolve_collision(self, tags: List[Tag], bit_range: Tuple[int, int] = None) -> Tuple[str, List[int]]:
        """处理一组标签对象的碰撞。"""
        raise NotImplementedError

    def resolve_collision_raw_strings(self, bit_strings: List[str]) -> Tuple[str, List[int]]:
        raise NotImplementedError

    def _apply_ber(self, bit_string: str, ber: float) -> str:
        """内部工具：应用比特误码率 (BER)。"""
        if ber <= 0:
            return bit_string
        noisy_bits = list(bit_string)
        for i in range(len(noisy_bits)):
            if random.random() < ber:
                noisy_bits[i] = '1' if noisy_bits[i] == '0' else '0'
        return "".join(noisy_bits)

    def _compute_ideal_mixed_signal(self, bit_strings: List[str]) -> Tuple[str, List[int]]:
        """内部工具：计算理想状态下的混合信号 (X)。"""
        if not bit_strings:
            return "", []

        length = len(bit_strings[0])
        signal = []
        collision_indices = []

        for i in range(length):
            bit_val = bit_strings[0][i]
            is_collision = False
            for j in range(1, len(bit_strings)):
                if bit_strings[j][i] != bit_val:
                    is_collision = True
                    break

            if is_collision:
                signal.append('X')
                collision_indices.append(i)
            else:
                signal.append(bit_val)

        return "".join(signal), collision_indices


class IdealChannel(BaseChannel):
    """
    理想信道：无丢包，无误码，无漏检。
    """

    def __init__(self):

        self.ber = 0.0
        self.miss_rate = 0.0
        self.dropout_rate = 0.0

    def resolve_collision(self, tags: List[Tag], bit_range: Tuple[int, int] = None) -> Tuple[str, List[int]]:
        if not tags:
            return "", []
        ids = [t.id for t in tags]
        if bit_range:
            start, end = bit_range
            ids = [s[start:end] for s in ids]
        return self._compute_ideal_mixed_signal(ids)

    def resolve_collision_raw_strings(self, bit_strings: List[str]) -> Tuple[str, List[int]]:
        """V3.1: 理想信道直接计算混合信号。"""
        return self._compute_ideal_mixed_signal(bit_strings)


class NoisyChannel(BaseChannel):
    """
    非理想信道：模拟真实物理层损伤。
    """

    def __init__(self, dropout_rate: float, miss_rate: float, ber: float):
        self.dropout_rate = dropout_rate
        self.miss_rate = miss_rate
        self.ber = ber

    def filter_active_tags(self, tags: List[Tag]) -> List[Tag]:
        """模拟标签概率性不响应。"""
        if self.dropout_rate <= 0:
            return tags
        return [t for t in tags if random.random() > self.dropout_rate]

    def resolve_collision(self, tags: List[Tag], bit_range: Tuple[int, int] = None) -> Tuple[str, List[int]]:
        """处理标签对象列表的碰撞。"""
        if not tags:
            return "", []

        ids = [t.id for t in tags]
        if bit_range:
            start, end = bit_range
            ids = [s[start:end] for s in ids]

        return self.resolve_collision_raw_strings(ids)

    def resolve_collision_raw_strings(self, bit_strings: List[str]) -> Tuple[str, List[int]]:
        if not bit_strings:
            return "", []

        noisy_strings = [self._apply_ber(s, self.ber) for s in bit_strings]

        signal_str, ideal_indices = self._compute_ideal_mixed_signal(
            noisy_strings)
        signal_list = list(signal_str)
        final_indices = []

        for idx in ideal_indices:
            if random.random() < self.miss_rate:

                signal_list[idx] = random.choice(['0', '1'])
            else:
                final_indices.append(idx)

        return "".join(signal_list), final_indices


class TraditionalAlgorithmInterface:
    def __init__(self, tags_in_field: List[Tag], **kwargs):
        self.tags_in_field = tags_in_field
        self.identified_tags: Set[str] = set()
        self.metrics = {'success_slots': 0,
                        'idle_slots': 0, 'collision_slots': 0}

        dropout = kwargs.get('dropout_rate', 0.0)
        miss_rate = kwargs.get('collision_miss_rate', 0.0)
        ber = kwargs.get('ber', 0.0)

        if dropout > 0 or miss_rate > 0 or ber > 0:
            self.channel = NoisyChannel(dropout, miss_rate, ber)
        else:
            self.channel = IdealChannel()

    def perform_step(self) -> AlgorithmStepResult:
        raise NotImplementedError

    def is_finished(self) -> bool:
        raise NotImplementedError

    def get_results(self) -> Set[str]:
        return self.identified_tags

    def get_active_tag_count(self) -> int:
        return len(self.tags_in_field) - len(self.identified_tags)


def generate_scenario(scenario_config: Dict) -> List[Tag]:
    """生成测试场景ID分布。"""
    total_tags = scenario_config.get('TOTAL_TAGS', 1000)
    binary_length = scenario_config.get('BINARY_LENGTH', 96)
    distribution_mode = scenario_config.get('id_distribution', 'random')

    id_set = set()

    if distribution_mode == 'sequential':
        for i in range(total_tags):
            id_set.add(format(i, f'0{binary_length}b'))
    elif distribution_mode == 'prefixed':
        prefix_len = scenario_config.get('prefix_length', binary_length // 2)
        prefix = ''.join(random.choice('01') for _ in range(prefix_len))
        while len(id_set) < total_tags:
            suffix = ''.join(random.choice('01')
                             for _ in range(binary_length - prefix_len))
            id_set.add(prefix + suffix)
    elif distribution_mode == 'dispersed':
        if total_tags > 1:
            required_positions = math.ceil(math.log2(total_tags)) + 2
        else:
            required_positions = 1
        base_id = ['0'] * binary_length
        start_pos = binary_length // 2
        possible_positions = list(range(start_pos, binary_length))
        num_positions_to_use = min(required_positions, len(possible_positions))
        collision_indices = random.sample(
            possible_positions, num_positions_to_use)
        while len(id_set) < total_tags:
            new_id_list = list(base_id)
            for index in collision_indices:
                new_id_list[index] = random.choice('01')
            id_set.add("".join(new_id_list))
    else:
        while len(id_set) < total_tags:
            id_set.add(''.join(random.choice('01')
                       for _ in range(binary_length)))

    return [Tag(identity_code=tag_id) for tag_id in id_set]


def apply_ber_noise(bit_string: str, ber: float) -> str:
    channel = BaseChannel()
    return channel._apply_ber(bit_string, ber)


def calculate_time_delta(step_result: AlgorithmStepResult, guard_interval_us: float = 0.0) -> float:
    """计算单步操作的时间开销。"""
    if step_result.override_time_us is not None:
        return step_result.override_time_us

    if step_result.operation_type == 'internal_op':
        return 0.0

    time_reader_tx = step_result.reader_bits / CONSTANTS.READER_BITS_PER_US

    base_time = 0.0
    if step_result.operation_type == 'idle_slot':
        base_time = time_reader_tx + 2.0 * CONSTANTS.T1_US
    elif step_result.operation_type in ['success_slot', 'collision_slot']:
        time_tag_response_window = step_result.expected_max_tag_bits / CONSTANTS.TAG_BITS_PER_US
        base_time = (time_reader_tx + CONSTANTS.T1_US +
                     time_tag_response_window + CONSTANTS.T2_MIN_US)
    else:
        return 0.0

    if step_result.operation_type in ['success_slot', 'collision_slot', 'idle_slot']:
        return base_time + guard_interval_us

    return base_time


def run_simulation(
    scenario_config: Dict,
    algorithm_class,
    algorithm_specific_config: Dict
) -> Dict:
    """运行单次仿真实验。"""
    enable_refined_energy = algorithm_specific_config.get(
        'enable_refined_energy_model', False)
    enable_monitoring = algorithm_specific_config.get(
        'enable_resource_monitoring', False)
    guard_interval = algorithm_specific_config.get('guard_interval_us', 0.0)

    result_dict = {
        'total_protocol_time_us': 0.0,
        'total_reader_bits': 0.0,
        'total_tag_bits': 0.0,
        'total_steps': 0,
        'total_listening_energy_uj': 0.0,
        'total_transmission_energy_uj': 0.0,
        'total_energy_uj': 0.0,
        'peak_metrics': {}
    }

    scenario_tags = generate_scenario(scenario_config)
    total_tags_in_scenario = len(scenario_tags)
    if total_tags_in_scenario == 0:
        return result_dict

    algo_instance = algorithm_class(scenario_tags, **algorithm_specific_config)

    max_steps = total_tags_in_scenario * 400
    step_count = 0

    while not algo_instance.is_finished():
        step_result = algo_instance.perform_step()

        time_delta = calculate_time_delta(step_result, guard_interval)

        result_dict['total_protocol_time_us'] += time_delta
        result_dict['total_reader_bits'] += step_result.reader_bits
        result_dict['total_tag_bits'] += step_result.tag_bits

        tx_energy_delta = (step_result.reader_bits * 2.0 +
                           step_result.tag_bits * 0.5) / 1000.0
        result_dict['total_transmission_energy_uj'] += tx_energy_delta

        active_tags = algo_instance.get_active_tag_count(
        ) if enable_refined_energy else total_tags_in_scenario
        listening_energy_delta = time_delta * active_tags * 0.1 / 1000.0
        result_dict['total_listening_energy_uj'] += listening_energy_delta

        if enable_monitoring and step_result.internal_metrics:
            for key, value in step_result.internal_metrics.items():
                current_peak = result_dict['peak_metrics'].get(
                    key, float('-inf'))
                result_dict['peak_metrics'][key] = max(current_peak, value)

        step_count += 1
        if max_steps > 0 and step_count > max_steps:
            print(f"错误: 仿真步骤 ({step_count}) 超过最大限制 ({max_steps})，强制终止。")
            break

    result_dict['total_steps'] = step_count
    result_dict['identified_tags_count'] = len(algo_instance.get_results())
    result_dict['total_energy_uj'] = result_dict['total_transmission_energy_uj'] + \
        result_dict['total_listening_energy_uj']

    if hasattr(algo_instance, 'metrics'):
        result_dict.update(algo_instance.metrics)

    if result_dict['total_protocol_time_us'] > 0:
        total_time_sec = result_dict['total_protocol_time_us'] / 1.0e6
        result_dict['throughput_tags_per_sec'] = result_dict['identified_tags_count'] / total_time_sec
    else:
        result_dict['throughput_tags_per_sec'] = 0.0

    return result_dict
