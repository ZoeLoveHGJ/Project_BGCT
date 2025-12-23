BGCT: Efficient RFID Tag Anti-Collision Algorithm Simulation Framework

📖 Introduction

This repository contains the official implementation of the BGCT (Bit-Group Collision Tree) algorithm. BGCT is a novel RFID tag anti-collision protocol designed to optimize identification efficiency and minimize energy consumption in large-scale tag environments.

In addition to the core BGCT implementation, this repository provides a comprehensive Discrete Event Simulation Framework written in Python. It simulates the communication between the RFID reader and tags. The project also includes reproductions of various state-of-the-art tree-based anti-collision algorithms as baselines, along with a complete suite of experimental scripts to verify performance under various conditions.

Key Features

** robust Simulation Kernel**: Built on Framework.py, supporting slot-level simulation, collision detection, and energy statistics.

Comprehensive Baselines: Includes implementations of DL-PCT, DQTA, EMDT, ICT, LAPCT, and more for fair comparison.

Rich Metrics: Automatically generates KPIs such as System Throughput, System Efficiency, Total Bits, and Energy Consumption.

Highly Configurable: Supports customization of tag quantities, ID lengths, ID distribution patterns, and Bit Error Rate (BER).

📂 Project Structure

Project_BGCT/
├── BGCT.py                 # Implementation of the proposed algorithm (BGCT)
├── BGCT_Random.py          # Randomized variant of BGCT
├── Framework.py            # Core RFID simulation framework
├── algorithm_base_config.py# Base configuration class for algorithms
├── Tool.py                 # Utility functions
│
├── baselines/              # (Logical Grouping) Comparison Algorithms
│   ├── DL_PCT_Final.py     # Dynamic Length Prefix Collision Tree
│   ├── DQTA.py             # Dynamic Quad Tree Algorithm
│   ├── EMDT.py             # Enhanced Multi-Dimension Tree
│   ├── ICT.py              # Improved Collision Tree
│   ├── LAPCT.py            # Look-Ahead Prefix Collision Tree
│   ├── NLHQT.py            # Non-Linear Hybrid Quad Tree
│   └── ... (Others: FHS_RAC, HT_EEAC, etc.)
│
├── experiments/            # (Logical Grouping) Experiment Scripts
│   ├── Exp0_d_target.py    # Exp 0: Parameter tuning (d_target)
│   ├── Exp1_Scalability.py # Exp 1: Performance vs. Number of Tags
│   ├── Exp2_Communication.py # Exp 2: Robustness against BER (Bit Error Rate)
│   ├── Exp3_Stability.py   # Exp 3: Impact of Tag ID Distributions
│   ├── Exp4_Ber.py         # Exp 4: Further channel interference tests
│   ├── Exp5_Distribution.py# Exp 5: Strategy analysis under distributions
│   └── Exp6_ComType.py     # Exp 6: Comparison of communication types
│
└── results/                # Output directory for .csv data and .png plots


🛠️ Requirements & Installation

This project is developed in Python. It is recommended to use Anaconda or a Python 3.8+ environment.

Clone the repository

git clone [https://github.com/ZoeLoveHGJ/Project_BGCT.git](https://github.com/ZoeLoveHGJ/Project_BGCT.git)
cd Project_BGCT


Install Dependencies
The project relies on numpy for calculations, and matplotlib/pandas for data analysis and plotting.

pip install numpy pandas matplotlib tqdm


🚀 Quick Start

1. Run a Single Algorithm

You can run any algorithm script directly to see the output of a single simulation session:

python BGCT.py


Output: Displays the number of slots, total bits, and runtime required to identify a specific number of tags.

2. Run Comparison Experiments

To reproduce the experimental results, run the scripts starting with Exp. For example, to test scalability across different numbers of tags:

python Exp1_Scalability.py


After execution, the results (CSV files) and performance plots (PNG images) will be automatically saved in the results/ directory.

📊 Experiments Description

The repository includes the following major experimental scenarios:

Script Name

Description

Key Variable

Exp1_Scalability

Scalability Test: Evaluates performance as tag quantity increases (e.g., 100 to 1000+).

Number of Tags

Exp2_Communication

Robustness Test: Evaluates algorithm stability under different channel Bit Error Rates (BER).

Bit Error Rate (BER)

Exp3_Stability

Distribution Test: Tests performance under Uniform, Consecutive, or Discrete ID distributions.

ID Distribution

Exp4_Ber

Parameter Tuning: Sensitivity analysis for internal BGCT parameters (e.g., d_target, d_max).

Algorithm Parameters

📝 Baselines Included

To ensure a fair evaluation of BGCT, we have reproduced the following classic and state-of-the-art tree-based algorithms:

QT 

DQTA

EMDT

ICT

LAPCT

NLHQT

SD-CGQT


Note: This code is for academic research purposes. If you use this code in your work, please cite the relevant publication.