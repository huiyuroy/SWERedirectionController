# SWERedirectionController

Return2MaxPotentialEnergy (R2MPE) is a redirection reset method that calculates optimal reset direction based on 
Probability Density Distribution (PDF). Generally, R2MPE can be combined with existing Redirected Walking controllers (e.g., 
S2C, S2O, ARC, APF-based, etc.). For more details, please refer to https://doi.org/10.1109/TVCG.2024.3409734



## How to use
We provide simulation 

1. download dataset.
2. place the project folder and the dataset folder in the same directory.
3. space data prepare:
   - run batch_process_rl_scenes.py 
   - run batch_process_scenes.py 
4. if the trajectories are not available
5. 
6. run swerc_simu_demo.py

## Dataset
To enable the simulation test/demo, the dataset is provided in:

[polyspaces]https://github.com/huiyuroy/polyspaces


## Requirement
- Python 3.12
- shapely
- pygame
- matplotlib
- ujson
- pytorch



