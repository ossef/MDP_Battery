# MDP Battery and Ngreen Models  

## I - Goal
 
This framework facilitates the resolution of large-scale stochastic decision-making problems, specifically Markov Decision Processes (MDPs) with a particular structure. Typically, the transition graph for each action exhibits this same structure; that is, all cycles within the graph pass through a common state. In our examples, this common state represents an empty container in the Green model or an empty battery in the Battery model. Details about the Green model can be found in [1], though it is discussed solely in the context of Markov chain modeling. However, the details of the Battery model are provided in [2], which indeed addresses an MDP problem for discretized energy packets where we seek finding the optimal policy to sell energy for an off-grid telecom operator.

## II - Project architecture

Tree of the most important files and folder in the project's repository :

```
/
├─┬─Models/: To store all MDP models (just a snapchot of models ... the original directory is about 18Go of data)
│ ├─- Ngreen/: To store all Ngreen MDP models (paper [1])
│ │     ├─-Model_N_50: Model with 50 states, taking maximum 100 actions
│ │     ├─-Model_N_500: Model with 500 states, taking maximum 100 actions
│ │     ├─- ...
│ │     └─ Model_N_100000: Model with 100 000 states, taking maximum 100 actions
│ └── Battery/: To store all Battery MDP models (paper [2])
│       ├─-One_Phase : different models for one phase battery : Model_B_6_N_14, Model_B_8_N_20, ... Model_B_1000_N_160000
│       ├─-Two_Phases_Scaling : different models for two phases battery : Model_B_15_N_100, Model_B_18_N_200, ... Model_B_700_N_200000  
│       └─-Two_Phases_Scenarios : different models for two phases battery : Model_B_6_N_30, Model_B_20_N_215  
│ 
├─┬─src/: Source folder.
│ ├─-Graph.py: Exhaustive search.
│ ├──Models.py: Greedy proposed method.
│ ├──AverageRewardFull.py: Local search.
│ ├──AverageRewardSparse.py: Local search.
│ │──Plots.py: Local search.
│ └──Launcher.py: Tabu search.
│ 
├─┬─ Results/: heatmap results of different Battery filling scenarios
│ ├─- Battery_Day_Night_215_r1_1_r3_-5000.pdf
│ ├─- Battery_Day_Night_215_r1_1_r3_-1000.pdf
│ ├─- Battery_Day_Night_215_r1_1_r3_-100.pdf
│ ├─- Battery_Day_Night_215_r1_1_r3_0.pdf
│ └── ... other experiments
│ 
├───.gitignore: To avoid junk files on git repository. 
└───README.md: This file.
```

## III - Build and run

Before running code, make sur you have these python dependencies for the "/Solve" directory :

    numpy, scipy, time, matplotlib

Uncomment and run function from Launcher.py : 

    python Launcher.py

## IV - Usage

Firstly, in the file "/Solve/Models.py", specify the model you want to use by modifying the value of "MDP_Case". MDP_Cases from 0 to 3 are automatically generated in the corresponding "If" structure. In contrast, large-scale models identified by MDP_Case from 4 to 7 have been generated separately using the <a href="https://github.com/ossef/XBorne-2017" target="_blank"> Xborne </a> tool in C language, and are stored in the "/Models" directory. Therefore, if you wish to test a large-scale model from 4 to 7, it is crucial to verify its presence in the directory. Additionally, you can create your own model, add it to "/Models", insert the corresponding "if" condition in "/Solve/Models.py", and update the file "/Solve/Launcher.py".Then, execute "Launcher.py", which will automatically generate the results of the chosen model.

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
    <img src="./screenshots/curves_uncomment_in_launcher.png" width="400" height="150"/>
</div>
<br>

#### Results for this model

There is only one test, but you can generate different types of graphs with different configurations. 
In particular, you can choose the number of wsm curves displayed, to start at the same point or not, to show all 
curves with the same scale and the position where the points should be selected.

<br>
<div align="center">
    <img src="./screenshots/curves_uncomment_in_test_1.png" width="400" height="250"/>
</div>
<br>

### B) Test 2 : Analyse of optimale policy for a specific scenario

Allows to show all algorithms already implemented in action.
To use this test, you need to uncomment the function `si.launch()` in `launcher.py`, after that, you can go to `single.py` 
to uncomment the actual function to use, moreover you can add or remove
your own test in this file.

#### Choosing a model, and a specific size 

<br>
<div align="center">
    <img src="./screenshots/single_uncomment_in_launcher.png" width="400" height="150"/>
</div>
<br>

#### Results for this model

Once your tests are finished, you can check in `resources/single/<method>/` to see the generated graphs and their data (respectively in pdf and csv files).
For example, here is the result of the exhaustive search :

<br>
<div align="center">
    <img src="./screenshots/single_exhaustive_results_1.png" width="500" height="125"/>
</div>
<br>
<div align="center">
    <img src="./screenshots/single_exhaustive_results_2.png" width="500" height="25"/>
</div>
<br>

## V - Illustration

In this section all screenshots related to the test are displayed here.

### A) Curves 

<div align="center">
    <img src="./screenshots/curves_example_1.png" width="300" height="250"/>
    <img src="./screenshots/curves_example_2.png" width="300" height="250"/>
    <img src="./screenshots/curves_example_3.png" width="300" height="250"/>
    <img src="./screenshots/curves_example_4.png" width="300" height="250"/>
</div>    

### B) Latex (example of a small instance)

<div align="center">
    <img src="./screenshots/latex_example_1.png" width="600" height="140"/>
    <img src="./screenshots/latex_example_2.png" width="600" height="70"/>
    <img src="./screenshots/latex_example_3.png" width="600" height="120"/>
    <img src="./screenshots/latex_example_4.png" width="600" height="50"/>

</div>



##  Contributors & Copyright

- [Youssef AIT EL MAHJOUB](https://github.com/ossef)
- This code is open source. However, one can cite the original document [2] submitted in WIMOB 2024.

[1] "Performance and energy efficiency analysis in NGREEN optical network", Youssef AIT EL MAHJOUB, Hind CASTEL-TALEB and Jean-Michel FOURNEAU". In, 14th International Conference on Wireless and Mobile Computing, Networking and Communications, WiMob, 2018.

[2] "Finding the optimal policy to sell energy for an off-grid telecom operator", Youssef AIT EL MAHJOUB and Jean-Michel FOURNEAU". Submitted paper, In, 20th International Conference on Wireless and Mobile Computing, Networking and Communications, WiMob, 2024.


