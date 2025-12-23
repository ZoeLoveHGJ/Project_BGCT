BGCT: 高效RFID标签防碰撞算法仿真框架

📖 项目简介

本项目是 BGCT (Bit-Group Collision Tree) 算法的官方实现代码库。BGCT 是一种新型的 RFID 标签防碰撞算法，旨在解决大规模标签环境下的识别效率和能量消耗问题。

本代码库不仅包含 BGCT 的核心实现，还提供了一个完整的离散事件仿真框架 (Framework)，用于模拟阅读器与标签之间的通信过程。此外，项目中复现了多种经典的树形及混合防碰撞算法作为对比基准（Baselines），并包含完整的实验脚本以验证算法在不同条件下的性能。

核心特性

完整的仿真内核：基于 Framework.py 构建，支持时隙模拟、碰撞检测及能量统计。

全面的对比实验：复现了包括 DL-PCT, DQTA, EMDT, ICT, LAPCT 等在内的多种前沿算法。

多维度评估指标：自动生成吞吐率 (Throughput)、系统效率 (System Efficiency)、通信开销 (Total Bits) 及能耗 (Energy) 等 KPI 报表。

高度可配置：支持自定义标签数量、ID长度、ID分布类型及信道误码率 (BER)。

📂 文件结构说明

Project_BGCT/
├── BGCT.py                 # 本项目提出的核心算法 (BGCT) 实现
├── BGCT_Random.py          # BGCT 算法的随机化变体
├── Framework.py            # RFID 通信仿真核心框架
├── algorithm_base_config.py# 算法基础配置类
├── Tool.py                 # 工具函数库
│
├── baselines/              # (逻辑分类) 对比算法实现
│   ├── DL_PCT_Final.py   
│   ├── DQTA.py           
│   ├── EMDT.py            
│   ├── ICT.py             
│   ├── LAPCT.py            
│   ├── NLHQT.py            
│   └── ... (其他对比算法: FHS_RAC, HT_EEAC 等)
│
├── experiments/            # (逻辑分类) 实验脚本
│   ├── Exp0_d_target.py    # 实验0：参数 d_target 调优
│   ├── Exp1_Scalability.py # 实验1：标签数量扩展性测试
│   ├── Exp2_Communication.py # 实验2：通信误码率 (BER) 鲁棒性测试
│   ├── Exp3_Stability.py   # 实验3：ID 分布稳定性测试
│   ├── Exp4_Ber.py         # 实验4：进一步的信道干扰测试
│   ├── Exp5_Distribution.py# 实验5：标签分布对策略的影响
│   └── Exp6_ComType.py     # 实验6：不同通信类型的对比
│
└── results/                # 实验结果输出目录 (.csv 数据及 .png 图表)


🛠️ 环境要求与安装

本项目基于 Python 开发。建议使用 Anaconda 或 Python 3.8+ 环境。

克隆仓库

git clone [https://github.com/your-username/Project_BGCT.git](https://github.com/your-username/Project_BGCT.git)
cd Project_BGCT


安装依赖
项目主要依赖 numpy 进行计算，matplotlib 和 pandas 用于数据处理与绘图。

pip install numpy pandas matplotlib tqdm


🚀 快速开始

1. 运行单个算法

你可以直接运行某个算法的脚本来查看单次仿真的输出：

python BGCT.py


输出示例：将显示识别特定数量标签所需的时隙数、通信比特数及运行时间。

2. 运行对比实验

复现论文中的实验结果，请运行 Exp 开头的脚本。例如，测试算法在不同标签数量下的扩展性：

python Exp1_Scalability.py


程序运行结束后，结果数据（CSV）和性能对比图（PNG）将自动保存在 results/ 目录下对应的子文件夹中。

📊 实验描述

本项目包含以下主要实验场景：

实验脚本

描述

关键变量

Exp1_Scalability

扩展性测试：评估算法在标签数量从 100 增加到 1000+ 时的性能变化。

标签数量 (Tag Num)

Exp2_Communication

鲁棒性测试：在不同信道误码率 (BER) 环境下评估算法的稳定性。

误码率 (BER)

Exp3_Stability

分布测试：测试标签 ID 在均匀分布、连续分布或特定离散分布下的表现。

ID 分布类型

Exp4_Ber

参数调优：针对 BGCT 算法内部参数 (如 d_target, d_max) 的敏感性分析。

算法内部参数

📝 包含的基准算法 (Baselines)

为了公平评估 BGCT 的性能，我们复现了以下经典与最新的树形防碰撞算法：

QT 

DQTA

EMDT

ICT

LAPCT

NLHQT

SD-CGQT

注：本代码仅供学术研究使用，如在论文中使用本代码，请引用相关发表文献。