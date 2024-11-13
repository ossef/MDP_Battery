# MDP Battery and Ngreen Models  

## I - Goal
 
This framework facilitates the resolution of large-scale stochastic decision-making problems, specifically for average reward criteria in unichain-type Markov Decision Processes (MDPs) with a particular structure. Typically, the transition graph for each action displays a unique characteristic: all cycles within the graph pass through a common state. In a previous model, the NGreen model presented in [1], this common state represents an empty optical container. In contrast, in [2] and [3], this state represents an empty battery in the Battery model. Details about the NGreen model can be found in [1], although it is discussed solely within the context of Markov chain modeling. Further details of the Battery model with ON and OFF energy duality are provided in [2], while a generic model with PV-ON and PV-OFF states, validated using real-world data (<a href="https://www.nrel.gov/research/areas.html" target="_blank"> NREL - National Renewable Energy Laboratory </a>), is presented in [3]. The studies in [2, 3] specifically address an MDP (Markov Decision Process) problem, making it highly relevant to this framework.
<br>
The Battery MDP model aims to identify the optimal policy for selling energy for an off-grid telecom operator. This involves balancing positive rewards from selling battery energy (and subsequently replacing it with a new empty one) against negative rewards, as the battery will no longer be available to power existing infrastructure. Additionally, there is a penalty for packet loss, i.e., when energy packets arrive while the current battery is full.

## II - Project architecture

Tree of the most important files and folder in the project's repository :

```
/
├─┬─Models/: To store all MDP models (just a snapchot of models ... the original directory is about 43Go)
│ ├─- Ngreen_Wimob2018/ : To store all Ngreen MDP models (paper [1])
│ │     ├─-Model_N_50     : Model with 50 states, taking maximum 100 actions
│ │     ├─-Model_N_500    : Model with 500 states, taking maximum 100 actions
│ │     ├─- ...
│ │     └─ Model_N_100000 : Model with 100 000 states, taking maximum 100 actions
│ ├─- Battery_Wimob2024/ : To store all Battery MDP models (paper [2])
│ │     ├─-One_Phase            : different models for one phase battery : Model_B_6_N_14, Model_B_8_N_20, ... Model_B_1000_N_160000
│ │     ├─-Two_Phases_Scaling   : different models for two phases battery : Model_B_15_N_100, Model_B_18_N_200, ... Model_B_700_N_200000  
│ │     └─-Two_Phases_Scenarios : different models for two phases battery : Model_B_6_N_30, Model_B_20_N_215  
│ └── Battery_ComCom2025/: To store all Battery MDP models (paper [3])
│       ├─-NREL_Data     : Raw data downloaded for different locations (Rabat, Paris, Barcelona, Moscow, Unalaska)
│       ├─-NREL_Extracts : Extracted discretized distributions from 'NREL_Data' for each city, by month
│       ├─-NREL_Models   : MDP models obtained from 'NREL_Extracts' distributions (refer to 'scriptMDP' file)
│       └─-Dist_gener.py : Code that transforms raw data from 'NREL_Data' into distributions in 'NREL_Extracts', making it usable by 'scriptMDP' in 'NREL_Models'
│
├─┬─Solve/: Source folder.
│ ├─-Graph.py               : reads Rii matrix storage from '/Models', converte to sparse_row.
│ ├──Models.py              : creates an MDP models, needs Graph.py to read external models 
│ ├──AverageRewardFull.py   : algorithms to solve average reward MDPs in Full storage
│ ├──AverageRewardSparse.py : algorithms to solve average reward MDPs in Sparse storage
│ │──Plots.py               : some util functions to draw 2d (and 3d) heatmaps, and comparison plots
│ └──Launcher.py            : The launcher programm
│ 
├─┬─ Results/: heatmap results of different Battery filling scenarios
│ ├─- WiMob2024/ :
│ │     └─-HeatMaps : Detailled Optimal policy for each state
│ │          ├─- Battery_Day_Night_215_r1_1_r3_-5000.pdf
│ │          ├─- Battery_Day_Night_215_r1_1_r3_-1000.pdf
│ │          ├─- Battery_Day_Night_215_r1_1_r3_0.pdf
│ │          └── ... other experiments
│ └─-ComCom2025/ :
│        └─- NREL : 
│             ├─- HeatMaps         : Detailled Optimal policy for each state, for a specific location and mounth
│             ├─- Rewards_Detailed : Average measures for different values of rewards r1, r2, r3 for a specific location and mounth
│             └── Cities_Months   : Optimal average measures (energy storage, Delay probability, energy loss). Different locations and mounths.
├───Screenshots/: contains some screenshots for below explanations
│
└───README.md: description file
```

## III - Build and run

Before running the code, ensure you have the following Python dependencies installed for the `Solve/` directory :

    numpy, scipy, time, matplotlib

Uncomment and run function from Launcher.py : 

    python3 Launcher.py

## IV - Usage

-Firstly, in the file `/Solve/Models.py`, specify the model you want to use by modifying the value of "MDP_Case". MDP_Cases from 0 to 3 are automatically generated in the corresponding "If" structure. In contrast, large-scale models identified by MDP_Case from 4 to 8 have been generated separately using the <a href="https://github.com/ossef/XBorne-2017" target="_blank"> Xborne </a> tool in C language, and are stored in the `/Models` directory. Therefore, if you wish to test a large-scale model from 4 to 8, it is crucial to verify its presence in the directory. Additionally, you can create your own model, add it to `/Models`, insert the corresponding "if" condition in `/Solve/Models.py`, and update the file `/Solve/Launcher.py`. Then, execute `/Solve/Launcher.py`, which will automatically generate the results of the chosen model.

-Please note that GitHub repository, can not support all referenced in Models.py. The complete set of analyzed models totals 43 GB. However, a small example is provided for each model. To adjust the model size and test the scalability of the algorithms, simply modify the parameters such as BufferSize, threshold, deadline, and actions in `./scriptMDP`. To alter the structure of the model, manipulation of <a href="https://github.com/ossef/XBorne-2017" target="_blank"> Xborne </a> is required, particularly with `fun.c` file, which encodes the structure of a Markov Chain, including the description of states, various events, transitions, and their probabilities.

In this project, there are two type of tests you can run :

### A) Test 1 : Numerical Comparison

Allows to display the execution time (in seconds), number of iteration, the average reward value
an the span value for different algorithms : 
-Relative Value Iteration (RVI)
-Natural Value Iteration (NVI)
-Relative Policy Iteration with Gauss-Jordan Elimination (RPI + GJ)
   (Evaluation phase is a direct method)
-Relative Policy Iteration with Fixed point method (RPI + FP)
   (Evaluation phase is an iterative method)
-Relative Policy Iteration with Rob-TypeB structure (RPI + RB)
   (Evaluation phase is a direct method)

#### Choosing a model, and a specific size 

<br>
<div align="center">
    <img src="./Screenshots/MDP_Case_6.png" width="800" height="600"/>
</div>
<br>

#### Results for this model

The test includes the five algorithms mentioned above. 
(Comment out any that you are not interested in)

<br>
<div align="center">
    <img src="./Screenshots/MDP_Case_6_1.png"" width="300" height="300"/>
    <img src="./Screenshots/MDP_CAse_6_2.png"" width="350" height="300"/>
</div>
<br>

### B) Test 2 : Analyse of optimale policy for a specific scenario

This allows to display the optimal policy for a specific scenario. It is not necessary to relaunch all five algorithms, using just one is sufficient to generate the optimal policy. RPI + RB is the fastest option

#### Choosing a WiMob2024 model, with a specific size 

<br>
<div align="center">
    <img src="./Screenshots/MDP_case_7.png" width="800" height="400"/>
</div>
<br>

#### Results for this model

Monitor reward values r1<sup>+</sup>, r2<sup>-</sup>, r3<sup>-</sup> in the MDP_Case = 7 section of `Solve/Launcher.py`. 
After completing your tests, you can find the generated PDF file in the `Results/WiMob2024/HeatMaps/` directory. 
For instance, the results with current rewards are stored in `Results/WiMob2024/HeatMaps/Battery_Day_Night_215_r1_1_r3_-5000.pdf`:

<br>
<div align="center">
    <img src="./Screenshots/MDP_Case_7_1.png" width="500" height="350"/>
    <img src="./Screenshots/MDP_Case_7_2.png" width="500" height="350"/>
</div>
<br>

#### Choosing a ComCom2025 model

In the MDP_Case = 8 section of `Solve/Launcher.py`, you can find three possible experiments. Uncomment the one you want to test: <br>
Experiment1 : Reward vs Measures — Provides results for different combinations of r1, r2, and r3. <br>
Experiment2 : Detailed Optimal Policy for a specific case — Gives the optimal policy for a specific combination of r1, r2, and r3. <br>
Experiment3 : Cities vs Months — Compares different locations and months of the year. <br>
In next, we test Experiment3
<br>
<div align="center">
    <img src="./Screenshots/MDP_Case_8.png" width="800" height="400"/>
</div>
<br>

#### Results for this model

Monitor reward values r1<sup>+</sup>, r2<sup>-</sup>, r3<sup>-</sup> in the MDP_Case = 8 section of `Solve/Launcher.py`. 
After completing your tests, you can find the generated PDF file in the `Results/ComCom2025/NREL/Cities_Months/` directory. 
For instance, the results with current rewards r1<sup>+</sup>=1, r2<sup>-</sup>=-100, r3<sup>-</sup>=-200 are stored in `Results/ComCom2025/NREL/Cities_Months/Cities_Months_Thr_25_B_65_r1_1_r2_-100_r3_-200`:

<br>
<div align="center">
    <img src="./Screenshots/MDP_Case_8_1.jpg" width="500" height="350"/>
    <img src="./Screenshots/MDP_Case_8_2.jpg" width="500" height="350"/>
    <img src="./Screenshots/MDP_Case_8_3.jpg" width="500" height="350"/>
    <img src="./Screenshots/MDP_Case_8_4.jpg" width="500" height="350"/>
</div>
<br>


##  Contributors & Copyright

- [Youssef AIT EL MAHJOUB](https://github.com/ossef)
- This code is open source. The original documents [1, 2, 3].

[1] "Performance and energy efficiency analysis in NGREEN optical network", Youssef AIT EL MAHJOUB, Hind CASTEL-TALEB and Jean-Michel FOURNEAU". In, 14th International Conference on Wireless and Mobile Computing, Networking and Communications, WiMob, 2018.

[2] "Finding the optimal policy to provide energy for an off-grid telecommunication operator", Youssef AIT EL MAHJOUB and Jean-Michel FOURNEAU". In, 20th International Conference on Wireless and Mobile Computing, Networking and Communications, WiMob, 2024.

[3] "A slot-based energy storage decision-making approach for optimal Off-Grid telecommunication operator", Youssef AIT EL MAHJOUB and Jean-Michel FOURNEAU". Pre-print submitted to Computer Communication Journal 2025, A special issue of WiMob2024. 


