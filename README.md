# USV_planner
Title: Local Collision Avoidance for Unmanned Surface Vehicles based on an End-to-End Planner with a LiDAR Beam Map

This paper has been accepted by IEEE Transactions on Intelligent Transportation Systems.  


>  **Introduction**: This paper eliminates the need for cumbersome map maintenance and complex feature extraction by directly translating sensor data into navigational actions. A key innovation is the "beam map"—a novel observation modality that detects obstacles in all directions, mimicking onboard LiDAR. To address collision avoidance maneuvers in various encounter situations, a continuous-time short-distance constraint algorithm is designed to calculate COLREGs compliance rewards. This enables legal and rational navigation without requiring prior knowledge of the encounter scenario. Extensive experimental results, comparing various RL policies and classical methods, demonstrate the planner’s exceptional obstacle avoidance capability and adaptability to changing environments.

### Beam map:

<img width="500" alt="beam-map" src="figure/beam-map.png" />



## Running
1. Create a new environment

```powershell
conda create -n usv_planner python=3.8
```
2. Install dependencies

```powershell
pip install -r requirements.txt
```
3. Build and install RVO2

Follow the steps from the official [github](https://github.com/rebuttal-anonymous/Python-RVO2)

5. Start to train USV_planner with TD3 policy

```powershell
python main_TD3.py
```

## Device
Ubuntu 20.04.6

## Result 

### 1. Trajectory comparison between different algorithms. 
Comparison algorithms:(a)DDPG (b)TD3 (c)DWA (d)APF  

<img width="500" alt="comparison" src="figure/comparison.png" />

### 2. Generalization ability verification.
<img width="700" alt="generalization" src="figure/generalization.png" />


## Acknowledgements
- The implementation of the beam map is inspired by the co-author [zw199502](https://github.com/zw199502/LSTM_EGO?tab=readme-ov-file#lstm_ego)'s code.

 - This project is based on [AntoineTheb](https://github.com/AntoineTheb/RNN-RL)'s implementation of the open-source solution of DRL structure.
 
 - The ORCA algorithm implementation is from [rebuttal-anonymous](https://github.com/rebuttal-anonymous/Python-RVO2)'s code.
  
 - The real inland AIS data is from [FVessel](https://github.com/gy65896/DeepSORVF).
 
 ## References:
 

> [1] W. Zhu and M. Hayashibe, "Learn to Navigate in Dynamic Environments with Normalized LiDAR Scans," 2024 IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan, 2024, pp. 7568-7575, doi: 10.1109/ICRA57147.2024.10611247. 


