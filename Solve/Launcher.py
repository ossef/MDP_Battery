from Models import *
import  AverageRewardSparse as sp
import  AverageRewardFull as fl
from Plots import *

if(MDP_Case<0 or MDP_Case>7):
    print("---------------------------------------")
    print("Error:  MDP_Case should be in {0,1,..7}")
    exit()

#------ Testing classical RVI and RPI algorithm on any MDP (Full Storage) -----#
if(MDP_Case == 0 or MDP_Case == 1):
    Ria, P, Ps, states = generate_MDP()
    moy1, policie1 = fl.relative_Value_Iteration_Matricielle(P, Ria, max_iter, epsilon, N, A)
    moy, policie   = fl.policy_Iteration_Matricielle_FP(P, Ria, max_iter, epsilon, N, A)
    moy2, policie2 = fl.policy_Iteration_Matricielle_GJ(P, Ria, max_iter, epsilon, N, A)

    #If the MDP has type B Robertazzi strucure, in full storage, on can uncommunt next line, 
    #moy0, policie0 = fl.policy_Iteration_Matricielle_ROB_B(Ps, Ria, max_iter, epsilon, N, A) 

    print("---------- Means rewards ---------")
    print("--> RVI ,moy = ",moy1)
    print("--> RPI+GJ ,moy = ",moy2)
    print("--> RPI+FP ,moy = ",moy)

    vals = np.array([moy,moy1,moy2])
    are_close = np.ptp(vals) <= 1e-8
    if(are_close == True or all(a == b == c   for a,b,c in zip(policie,policie1,policie2)) ):
        print("--------> Similarité OK !")
    else :
        print("--------> Not similaire !")

#------ Testing property of Robertazzi algorithms "The specific example in type B, Rob90 book", random python generation for rapid testing, (Sparse storage)  -----#
if(MDP_Case == 2):
    Ria, P, Ps, states = generate_MDP()
    moy0, policie0 = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
    moy1, policie1 = sp.relative_Value_Iteration_Csr(Ps, Ria, max_iter, epsilon, N, A)
    moy, policie   = sp.natural_Value_Iteration_Csr(Ps, Ria, max_iter, epsilon, N, A)
    moy2, policie2 = sp.policy_Iteration_Csr_GJ(Ps, Ria, max_iter, epsilon, N, A)
    moy3, policie3 = sp.policy_Iteration_Csr_FP(Ps, Ria, max_iter, epsilon, N, A)

    print("---------- Means rewards ---------")
    print("--> RPI+ROB ,moy = ",moy0)
    print("--> RVI ,moy = ",moy1)
    print("--> NVI ,moy = ",moy)
    print("--> RPI+GJ ,moy = ",moy2)
    print("--> RPI+FP ,moy = ",moy3)

    vals = np.array([moy0,moy1])
    are_close = np.ptp(vals) <= 1e-8
    if(are_close == True or all(a == b == c == d == e   for a,b,c,d,e in zip(policie,policie0,policie1,policie2,policie3)) ):
        print("--------> Similarité OK !")
    else :
        print("--------> Not similaire !")

#------ Testing scalability and precision of algorithms "type-B model", random python generation for rapid testing, (Sparse storage)  -----#
if(MDP_Case == 3):
    Ria, P, Ps, states = generate_MDP()
    moy0, policie0 = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
    moy1, policie1 = sp.relative_Value_Iteration_Csr(Ps, Ria, max_iter, epsilon, N, A)
    moy, policie   = sp.natural_Value_Iteration_Csr(Ps, Ria, max_iter, epsilon, N, A)
    moy2, policie2 = sp.policy_Iteration_Csr_GJ(Ps, Ria, max_iter, epsilon, N, A)
    moy3, policie3 = sp.policy_Iteration_Csr_FP(Ps, Ria, max_iter, epsilon, N, A)

    print("---------- Means rewards ---------")
    print("--> RPI+ROB ,moy = ",moy0)
    print("--> RVI ,moy = ",moy1)
    print("--> NVI ,moy = ",moy)
    print("--> RPI+GJ ,moy = ",moy2)
    print("--> RPI+FP ,moy = ",moy3)

    vals = np.array([moy,moy0,moy1,moy2,moy3])
    are_close = np.ptp(vals) <= 1e-8
    if(are_close == True or all(a == b == c == d == e   for a,b,c,d,e in zip(policie,policie0,policie1,policie2,policie3)) ):
        print("--------> Similarité OK !")
    else :
        print("--------> Not similaire !")

#------ Testing scalability and precision of algorithms, "Ngreen optical container", from Xborne tool, (Sparse storage) -----#
if(MDP_Case == 4):
    Ria, P, Ps, states = generate_MDP()
    moy0, policie0 = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
    moy1, policie1 = sp.relative_Value_Iteration_Csr(Ps, Ria, max_iter, epsilon, N, A)
    moy, policie   = sp.natural_Value_Iteration_Csr(Ps, Ria, max_iter, epsilon, N, A)
    moy2, policie2 = sp.policy_Iteration_Csr_GJ(Ps, Ria, max_iter, epsilon, N, A)
    moy3, policie3 = sp.policy_Iteration_Csr_FP(Ps, Ria, max_iter, epsilon, N, A)

    print("---------- Means rewards ---------")
    print("--> RPI+ROB ,moy = ",moy0)
    print("--> RVI ,moy = ",moy1)
    print("--> NVI ,moy = ",moy)
    print("--> RPI+GJ ,moy = ",moy2)
    print("--> RPI+FP ,moy = ",moy3)

    vals = np.array([moy,moy0,moy1,moy2,moy3])
    are_close = np.ptp(vals) <= 1e-8
    if(are_close == True or all(a == b == c == d == e   for a,b,c,d,e in zip(policie,policie0,policie1,policie2,policie3)) ):
        print("--------> Similarité OK !")
    else :
        print("--------> Not similaire !")

#------ Testing scalability and precision of algorithms, "an old Rob-B model with one phase", from Xborne tool, (Sparse storage) -----#
if(MDP_Case == 5):
    Ria, P, Ps, states = generate_MDP()
    moy0, policie0 = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
    moy1, policie1 = sp.relative_Value_Iteration_Csr(Ps, Ria, max_iter, epsilon, N, A)
    moy, policie   = sp.natural_Value_Iteration_Csr(Ps, Ria, max_iter, epsilon, N, A)
    moy2, policie2 = sp.policy_Iteration_Csr_GJ(Ps, Ria, max_iter, epsilon, N, A)
    moy3, policie3 = sp.policy_Iteration_Csr_FP(Ps, Ria, max_iter, epsilon, N, A)

    print("---------- Means rewards ---------")
    print("--> RPI+ROB ,moy = ",moy0)
    print("--> RVI ,moy = ",moy1)
    print("--> NVI ,moy = ",moy)
    print("--> RPI+GJ ,moy = ",moy2)
    print("--> RPI+FP ,moy = ",moy3)

    vals = np.array([moy,moy0,moy1,moy2,moy3])
    are_close = np.ptp(vals) <= 1e-8
    if(are_close == True or all(a == b == c == d == e   for a,b,c,d,e in zip(policie,policie0,policie1,policie2,policie3)) ):
        print("--------> Similarité OK !")
    else :
        print("--------> Not similaire !")

#------ Testing scalability and precision of algorithms, "WIMOB 2024" model, from Xborne tool, (Sparse storage) -----#
if(MDP_Case == 6):  
    Ria, P, Ps, states = generate_MDP()
    moy0, policie0 = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
    moy1, policie1 = sp.relative_Value_Iteration_Csr(Ps, Ria, max_iter, epsilon, N, A)
    moy, policie   = sp.natural_Value_Iteration_Csr(Ps, Ria, max_iter, epsilon, N, A)
    moy2, policie2 = sp.policy_Iteration_Csr_GJ(Ps, Ria, max_iter, epsilon, N, A)
    moy3, policie3 = sp.policy_Iteration_Csr_FP(Ps, Ria, max_iter, epsilon, N, A)

    print("---------- Means rewards ---------")
    print("--> RPI+ROB ,moy = ",moy0)
    print("--> RVI ,moy = ",moy1)
    print("--> NVI ,moy = ",moy)
    print("--> RPI+GJ ,moy = ",moy2)
    print("--> RPI+FP ,moy = ",moy3)

    vals = np.array([moy,moy0,moy1,moy2,moy3])
    are_close = np.ptp(vals) <= 1e-8
    if(are_close == True or all(a == b == c == d == e   for a,b,c,d,e in zip(policie,policie0,policie1,policie2,policie3)) ):
        print("--------> Similarité OK !")
    else :
        print("--------> Not similaire !")

#------ HeatMap and analyse for a specific scenario, "WIMOB 2024" model, from Xborne tool, (Sparse storage)   -----#
if(MDP_Case == 7): 
    r1 = 1      #Reward for battery release, fixé à 5, ok 
    r2 = 0      #Penalty packets lost, (pas d'influence), à re-écrire
    r3 = -5000      #Penalty packets delay, à re-écrire
    r4 = 0      #Penalty loop at (0,0,1), à re-écrire

    Ria, P, Ps, states = generate_MDP([], [], r1, r2, r3, r4, 0) 

    for r2 in range(0,-201, -100):
        print('',r1,', ',r2,', ',r3, ', ',r4)
        Ria, P, Ps, states = generate_MDP(Ps, states, r1, r2, r3, r4, 1)
        moy, policie = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
        plot_optimal_policy_2d(states, policie, A, seuil, r1, r2, r3, moy)

''' max = 5
    data = []
    Ria, P, Ps, states = generate_MDP([], [], 0, 0, 0, 0, 0)
    for r1 in range(0,max,1):
        for r2 in range(0, -201, -50):
            for r3 in range(0, -201, -100):
                print('',r1,', ',r2,', ',r3)
                Ria, P, Ps, states = generate_MDP(Ps, states, r1, r2, r3, r4, 1)
                moy, policie = policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
                data.append([r1, r2, r3, moy])
    plot_optimal_rewards_3d(data)'''
    #plot_optimal_policy_2d(states, policie, A, seuil)