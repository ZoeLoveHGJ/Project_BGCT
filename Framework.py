
import random
import math 
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional


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
    NAK_CMD_BITS: int = 8
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




class TraditionalAlgorithmInterface:
    def __init__(self, tags_in_field: List[Tag], **kwargs):
        self.tags_in_field = tags_in_field
        self.identified_tags: Set[str] = set()
        self.metrics = {'success_slots': 0, 'idle_slots': 0, 'collision_slots': 0}
        self.ber = kwargs.get('ber', 0.0)

    def perform_step(self) -> AlgorithmStepResult:
        raise NotImplementedError

    def is_finished(self) -> bool:
        raise NotImplementedError

    def get_results(self) -> Set[str]:
        return self.identified_tags

    def get_active_tag_count(self) -> int:
        return len(self.tags_in_field) - len(self.identified_tags)





def generate_scenario(scenario_config: Dict) -> List[Tag]:
    total_tags = scenario_config.get('TOTAL_TAGS', 1000)
    binary_length = scenario_config.get('BINARY_LENGTH', 96)
    distribution_mode = scenario_config.get('id_distribution', 'random')
    
    id_set = set()
    if distribution_mode == 'sequential':
        for i in range(total_tags):
            id_set.add(format(i, f'0{binary_length}b'))
    elif distribution_mode == 'prefixed':
        prefix_len = scenario_config.get('prefix_length', binary_length // 2)
        suffix_len = binary_length - prefix_len
        prefix = ''.join(random.choice('01') for _ in range(prefix_len))
        while len(id_set) < total_tags:
            suffix = ''.join(random.choice('01') for _ in range(suffix_len))
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
        collision_indices = random.sample(possible_positions, num_positions_to_use)

        while len(id_set) < total_tags:
            new_id_list = list(base_id)
            
            for index in collision_indices:
                new_id_list[index] = random.choice('01')
            id_set.add("".join(new_id_list))
    
            
    else: 
        while len(id_set) < total_tags:
            id_set.add(''.join(random.choice('01') for _ in range(binary_length)))
            
    return [Tag(identity_code=tag_id) for tag_id in id_set]

def apply_ber_noise(bit_string: str, ber: float) -> str:
    """独立的BER信道模型工具函数。"""
    if ber <= 0:
        return bit_string
    noisy_bits = list(bit_string)
    for i in range(len(noisy_bits)):
        if random.random() < ber:
            noisy_bits[i] = '1' if noisy_bits[i] == '0' else '0'
    return "".join(noisy_bits)

def calculate_time_delta(step_result: AlgorithmStepResult) -> float:
    """计算单步操作的时间开销。"""
    if step_result.override_time_us is not None:
        return step_result.override_time_us
    if step_result.operation_type == 'internal_op':
        return 0.0
    time_reader_tx = step_result.reader_bits / CONSTANTS.READER_BITS_PER_US
    if step_result.operation_type == 'idle_slot':
        return time_reader_tx + 2.0 * CONSTANTS.T1_US
    elif step_result.operation_type in ['success_slot', 'collision_slot']:
        time_tag_response_window = step_result.expected_max_tag_bits / CONSTANTS.TAG_BITS_PER_US
        return (time_reader_tx + CONSTANTS.T1_US + 
                time_tag_response_window + CONSTANTS.T2_MIN_US)
    else:
        print(f"警告: 未知的操作类型 '{step_result.operation_type}'，无法计算时间。")
        return 0.0

def run_simulation(
    scenario_config: Dict,
    algorithm_class,
    algorithm_specific_config: Dict
) -> Dict:
    enable_refined_energy = algorithm_specific_config.get('enable_refined_energy_model', False)
    enable_monitoring = algorithm_specific_config.get('enable_resource_monitoring', False)

    result_dict = {
        'total_protocol_time_us': 0.0, 'total_reader_bits': 0.0,
        'total_tag_bits': 0.0, 'total_steps': 0,
        'total_listening_energy_uj': 0.0,
        'total_transmission_energy_uj': 0.0,
        'total_energy_uj': 0.0,
        'peak_metrics': {} 
    }
    
    scenario_tags = generate_scenario(scenario_config)
    total_tags_in_scenario = len(scenario_tags)
    if total_tags_in_scenario == 0: return result_dict
    
    algo_instance = algorithm_class(scenario_tags, **algorithm_specific_config)
    
    max_steps = total_tags_in_scenario * 400 
    step_count = 0

    while not algo_instance.is_finished():
        step_result = algo_instance.perform_step()
        time_delta = calculate_time_delta(step_result)
        
        result_dict['total_protocol_time_us'] += time_delta
        result_dict['total_reader_bits'] += step_result.reader_bits
        result_dict['total_tag_bits'] += step_result.tag_bits
        
        
        
        tx_energy_delta = (step_result.reader_bits * 2.0 + step_result.tag_bits * 0.5) / 1000.0 
        result_dict['total_transmission_energy_uj'] += tx_energy_delta
        
        active_tags = algo_instance.get_active_tag_count() if enable_refined_energy else total_tags_in_scenario
        listening_energy_delta = time_delta * active_tags * 0.1 / 1000.0 
        result_dict['total_listening_energy_uj'] += listening_energy_delta
        
        if enable_monitoring and step_result.internal_metrics:
            for key, value in step_result.internal_metrics.items():
                current_peak = result_dict['peak_metrics'].get(key, float('-inf'))
                result_dict['peak_metrics'][key] = max(current_peak, value)
        
        step_count += 1
        if max_steps > 0 and step_count > max_steps:
            print(f"错误: 仿真步骤 ({step_count}) 超过最大限制 ({max_steps})，强制终止。")
            break

    result_dict['total_steps'] = step_count
    result_dict['identified_tags_count'] = len(algo_instance.get_results())
    result_dict['total_energy_uj'] = result_dict['total_transmission_energy_uj'] + result_dict['total_listening_energy_uj']
    
    if hasattr(algo_instance, 'metrics'):
        result_dict.update(algo_instance.metrics)

    if result_dict['total_protocol_time_us'] > 0:
        total_time_sec = result_dict['total_protocol_time_us'] / 1.0e6
        result_dict['throughput_tags_per_sec'] = result_dict['identified_tags_count'] / total_time_sec
    else:
        result_dict['throughput_tags_per_sec'] = 0.0
        
    return result_dict

