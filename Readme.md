# SWERedirectionController

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTVCG.2025.3595181-blue)](https://doi.org/10.1109/TVCG.2025.3595181)

A Deep Reinforcement Learning (DRL) based spatial walkability-aware redirection controller that guides users toward safer and more walkable physical areas using the innovative Spatial Walkability Entropy (SWE) metric, effectively enhancing obstacle avoidance capabilities in virtual reality environments.

> Li, H., Ding, Y., He, Y., Fan, L., & Xu, X. (2025). Towards Walkable and Safe Areas: DRL-Based Redirected Walking Leveraging Spatial Walkability Entropy. *IEEE Transactions on Visualization and Computer Graphics*.

## Key Features

- **Spatial Walkability Entropy (SWE)**: Novel metric quantifying navigation freedom and safety in physical space
- **DRL-based Controller**: Proximal Policy Optimization (PPO) agent with joint reward function
- **Reset Optimization**: Regional entropy maximization strategy reducing re-collision risk
- **Superior Performance**: Reduces physical collisions by 13.3% and extends walking distance by 12.6% vs SOTA
- **VR Integration**: Compatible with HMD systems like PICO 4 Pro

## How to Use

### 1. Download Dataset
Clone the Polyspaces dataset repository:
```bash
git clone https://github.com/huiyuroy/polyspaces
```

### 2. Project Setup
Place the project folder and dataset in the same directory:
```
your_directory/
├── SWERedirectionController/
└── polyspaces/
```

### 3. Space Data Preparation
Preprocess spatial data for training and simulation:
```bash

# Process RL scenes
run batch_process_rl_scenes.py

# Process general scenes
run batch_process_scenes.py
```

### 4. Run Simulation Demo
Execute the redirection controller simulation:
```bash
run swerc_simu_demo.py
```

### 5. Train Custom Model (Optional)
Train with custom configurations:
```bash
run swerc_trainer.py
```

## Dataset

The [Polyspaces Dataset](https://github.com/huiyuroy/polyspaces) contains 57 distinct virtual spatial layouts designed based on:
- Area size
- Boundary regularity
- Number of internal obstacles


## Requirements

- Python 3.12
- Core dependencies:
  ```bash
  pip install torch==2.1.0 shapely==2.0.2 pygame==2.5.0 matplotlib==3.8.0 ujson==5.9.0
  ```
- Full requirements: [requirements.txt](requirements.txt)


## Citation

If you use this work in your research, please cite:
```bibtex
@article{li2025towards,
  title={Towards Walkable and Safe Areas: DRL-Based Redirected Walking Leveraging Spatial Walkability Entropy},
  author={Li, Huiyu and Ding, Ying and He, Yuang and Fan, Linwei and Xu, Xiang},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2025},
  doi={10.1109/TVCG.2025.3595181}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact

- Project Lead: Huiyu Li (huiyuroy@163.com)
- Corresponding Author: Linwei Fan (lwfan129@163.com)
- Affiliation: Shandong University of Finance and Economics