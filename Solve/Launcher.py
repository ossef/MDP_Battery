from Models import *
import itertools
import  AverageRewardSparse as sp
import  AverageRewardFull as fl
from Plots import *

if(MDP_Case<0 or MDP_Case>8):
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
    r1 = 1          #Reward for battery release, fixé à 5, ok 
    r2 = -100       #Penalty packets lost, (pas d'influence)
    r3 = -5000      #Penalty packets delay
    r4 = 0          #Penalty loop at (0,0,1)

    Ria, P, Ps, states = generate_MDP([], [], r1, r2, r3, r4, 0) 
    print('',r1,', ',r2,', ',r3, ', ',r4)
    moy, policie = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
    plot_optimal_policy_2d_WiMob24(states, policie, A, seuil, r1, r2, r3, moy)


    """max = 5
    data = []
    Ria, P, Ps, states = generate_MDP([], [], 0, 0, 0, 0, 0)
    for r1 in range(0,max,1):
        for r2 in range(0, -201, -50):
            for r3 in range(0, -5000, -1000):
                print('',r1,', ',r2,', ',r3)
                Ria, P, Ps, states = generate_MDP(Ps, states, r1, r2, r3, r4, 1)
                moy, policie = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
                data.append([r1, r2, r3, moy])
    plot_optimal_rewards_3d(data)"""
    #plot_optimal_policy_2d(states, policie, A, seuil)

#------ HeatMap and analyse for a Real NSRDB Data sets, "ComCom 2025" model, from Xborne tool, (Sparse storage)   -----#
if(MDP_Case == 8): 

    DATA_TYPE = 2  # 1 for "NSRDB" data (only based on GHI indicator, not precise ! )
                   # 2 for "NREL" data  (based on meteorological indicators and PV-panel configurations, precise ! )

    if DATA_TYPE == 1 :
        r1 = 10         #Reward for battery release, fixé à 5, ok 
        r2 = -1e4       #Penalty packets lost, (pas d'influence)
        r3 = -100      #Penalty packets delay
        r4 = 0          #Penalty loop at (0,0,1)

        cityNames  = ["Chicago_Ohare", "Fairbanks", "Los_Angeles", "New_York_JFK", "Reno"] # "Fairbanks", "Reno"]
        years = [str(year) for year in range(1991, 2010 + 1, 1)]

        city, year = "Fairbanks", "2010"
        #N, n_packets, packet_size, hours_packets, pService, Buffer, Thr, h_deb, Deadline, Ria, P, Ps, states = generate_MDP([], [], r1, r2, r3, r4, 0, city, month, DATA_TYPE)

        N, n_packets, packet_size, hours_packets, pService, Buffer, Thr, h_deb, Deadline, Ria, P, Ps, states = generate_MDP([], [], r1, r2, r3, r4, 0, city, year, DATA_TYPE)
        moy, policie = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
        #energy, loss, noService = sp.average_Measures(Ps, states, policie, N, Buffer, Thr, h_deb, Deadline, pRelease, pService, n_packets, hours_packets)
        #print("Values : ",  energy, loss, noService, moy)
        plot_optimal_policy_2d_ComCom25(states, policie, A, Thr, Buffer, year, city, h_deb, Deadline, r1, r2, r3, moy, DATA_TYPE)

        """
        # Dictionnaire pour stocker les rewards moyennes
        rewards_data = {city: [] for city in cityNames}
        energy_data  = {city: [] for city in cityNames}
        loss_data    = {city: [] for city in cityNames}
        noService_data  = {city: [] for city in cityNames}

        for city in cityNames: 
            for year in years : 
                N, n_packets, packet_size, hours_packets, pService, Buffer, Thr, h_deb, Deadline, Ria, P, Ps, states = generate_MDP([], [], r1, r2, r3, r4, 0, city, year, DATA_TYPE)
                moy, policie = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
                energy, loss, noService = sp.average_Measures(Ps, states, policie, N, packet_size, Buffer, Thr, h_deb, Deadline, pRelease, pService, n_packets, hours_packets)
                moy = energy*r1 + loss*r2 + noService*r3
                rewards_data[city].append(moy)
                energy_data[city].append(energy)
                loss_data[city].append(loss)
                noService_data[city].append(noService)
                #plot_optimal_policy_2d(states, policie, A, Thr, Buffer, year, city, h_deb, Deadline, r1, r2, r3, moy)

        plot_cities_years_ComCom25(rewards_data, energy_data, loss_data, noService_data, cityNames, years, r1, r2, r3, Thr, Buffer, DATA_TYPE)
        """

    if DATA_TYPE == 2 :

        #---------------- Experiment1 : Reward vs Measures --------------- 

        """
        r1 = 1          #(1)   Reward for battery release, fixé à 5, ok 
        r4 = 0          #Penalty loop at (0,0,1)
        r2_values = [0,-100]#[0, -1, -10, -100, -300] #[0, -10, -100, -200, -300, -400, -500]         #[0, -1, -10, -100, -300] # Par exemple, les valeurs de r1 à tester
        r3_values = [0, -25]#[0, -1, -10, -20, -30, -50] #[0, -10, -50, -100, -200, -400 ] # Par exemple, les valeurs de r2 à tester

        city, month = "Barcelona", "M8"
        data_energy = []
        data_loss = []
        data_noService = []
        data_moy = []
        read = False

        for r2, r3 in itertools.product(r2_values, r3_values):
            print("\n------------------ Calcul pour r1={}, r2={}, r3={} ------------------".format(r1, r2, r3))
            if read == False:
                N, n_packets, packet_size, hours_packets, pService, Buffer, Thr, h_deb, Deadline, Ria, P, Ps, states = generate_MDP([], [], r1, r2, r3, r4, 0, city, month, DATA_TYPE)
                read = True
            else :
                Ria = generate_Reward(N, Thr, Buffer, h_deb, Deadline, packet_size, n_packets, hours_packets, Ps, states, r1, r2, r3, r4)          #Defined rewards

            # Appliquer l'algorithme d'itération de politique
            moy, policie = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)

            # Calculer les mesures moyennes
            energy, loss, noService = sp.average_Measures(Ps, states, policie, N, packet_size, Buffer, Thr, h_deb, Deadline, pRelease, pService, n_packets, hours_packets)
            moy = energy*r1 + loss*r2 + noService*r3

            # Stocker les résultats dans les listes de données
            data_energy.append([r2, r3, energy])
            data_loss.append([r2, r3, loss])
            data_noService.append([r2, r3, noService])
            data_moy.append([r2, r3, moy])
            plot_optimal_policy_2d_ComCom25(states, policie, A, Thr, Buffer, month, city, h_deb, Deadline, r1, r2, r3, moy, DATA_TYPE)



        plot_optimal_rewards_3d_ComCom25(data_energy, city, r1, month, 'Energy vs Rewards,  r1={}'.format(r1), 'Stored energy (Wh)')
        plot_optimal_rewards_3d_ComCom25(data_noService, city, r1, month, 'Delay vs Rewards, r1={}'.format(r1), 'Delay probability')
        plot_optimal_rewards_3d_ComCom25(data_loss, city, r1, month, 'Loss vs Rewards, r1={}'.format(r1), 'Energy loss (Wh)')
        plot_optimal_rewards_3d_ComCom25(data_moy, city, r1, month, 'Combined rewards vs Rewards, r1={}'.format(r1), 'Combined reward')
        """


        #---------------- Experiment2 : Detailed Optimal Policy for a specific case  ---------------

        """
        r1 = 1          #(1)   Reward for battery release, fixé à 5, ok 
        r4 = 0          #Penalty loop at (0,0,ON) (not used)
        r2 = -150         #(-1e10) Penalty packets lost
        r3 = -300          #(-e4)  Penalty packets delay

        city, month = "Barcelona", "M8"

        print("\n --------------------------- {}_{} -----------------------------".format(city,month))
        N, n_packets, packet_size, hours_packets, pService, Buffer, Thr, h_deb, Deadline, Ria, P, Ps, states = generate_MDP([], [], r1, r2, r3, r4, 0, city, month, DATA_TYPE)
        moy, policie = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
        energy, loss, noService = sp.average_Measures(Ps, states, policie, N, packet_size, Buffer, Thr, h_deb, Deadline, pRelease, pService, n_packets, hours_packets)
        #print("energy = {}, loss = {}, noService={}, moy = {}".format(energy, loss, noService, moy))
        moy = energy*r1 + loss*r2 + noService*r3
        plot_optimal_policy_2d_ComCom25(states, policie, A, Thr, Buffer, month, city, h_deb, Deadline, r1, r2, r3, moy, DATA_TYPE)
        """

        #---------------- Experiment3 : Cities vs Months --------------- 

        r1 = 1          #(1)   Reward for battery release, fixé à 5, ok 
        r4 = 0          #Penalty loop at (0,0,ON) (not used)
        r2 = -100         #(-1e10) Penalty packets lost
        r3 = -200          #(-e4)  Penalty packets delay

        cityNames = ["Unalaska", "Moscow", "Paris", "Barcelona", "Rabat" ]
        months   = ["M"+str(month) for month in range(1, 13, 1)]
        rewards_data = {city: [] for city in cityNames}
        energy_data  = {city: [] for city in cityNames}
        loss_data    = {city: [] for city in cityNames}
        noService_data  = {city: [] for city in cityNames}

        for city in cityNames: 
            for  month in months : 
                print("\n --------------------------- {}_{} -----------------------------".format(city,month))
                N, n_packets, packet_size, hours_packets, pService, Buffer, Thr, h_deb, Deadline, Ria, P, Ps, states = generate_MDP([], [], r1, r2, r3, r4, 0, city, month, DATA_TYPE)
                moy, policie = sp.policy_Iteration_Csr_ROB_B(Ps, Ria, max_iter, epsilon, N, A)
                energy, loss, noService = sp.average_Measures(Ps, states, policie, N, packet_size, Buffer, Thr, h_deb, Deadline, pRelease, pService, n_packets, hours_packets)
                moy = energy*r1 + loss*r2 + noService*r3
                rewards_data[city].append(moy)
                energy_data[city].append(energy)
                loss_data[city].append(loss)
                noService_data[city].append(noService)
                #plot_optimal_policy_2d(states, policie, A, Thr, Buffer, month, city, h_deb, Deadline, r1, r2, r3, moy, DATA_TYPE)

        plot_cities_years_ComCom25(rewards_data, energy_data, loss_data, noService_data, cityNames, months, r1, r2, r3, Thr, Buffer, DATA_TYPE)
