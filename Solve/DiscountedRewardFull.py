from Models import * 


def natural_Value_Iteration_Detailed(P, Ria, max_iter, epsilon, discount):
    # Algorithme de Value Iteration pour Discounted Reward
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@ Natural Value Iteration NVI-DR: Detailed version  @@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
    
    profiler = cProfile.Profile() #For processing time
    profiler.enable()             #Démarrez le profilage
    
    # Initialisation de la valeur moyenne pour chaque état
    J = [0 for _ in range(N)]
    Optimal_Policy = [0 for _ in range(N)]
    
    k = 0
    #print("---> Iteration k = ",k)
    #print("J = ",J)

    while(k <= max_iter):
        k += 1
        #print("---> Iteration k = ",k)
        J_prev = np.copy(J)

        #Calule de la nouvelle Q valeur, sur chaque etat 
        for s in range(N):
            Q = {}
            for a in range(A):
                Q[a] = Ria[s, a] + discount*sum(P[a,s,s_next]*J_prev[s_next] for s_next in range(N))
            #print("Q = ",Q)

            tmp = list(Q.values())
            a_opt = np.argmax(tmp)
            Optimal_Policy[s]=a_opt
            J[s] = tmp[a_opt]

        diff = J - J_prev
        norme = np.linalg.norm(diff, np.inf)
        #print("J = {}, norme = {:.15e} ".format([round(x,10) for x in J],norme))
        #print("Optimal Policy = ",Optimal_Policy)

        precision = epsilon*0.5*((1-discount)/discount) #0.000125 if epsilon = 1e-3 and discount = 0.8 (see Abhijit book)
        if norme < precision :
            break

    print("\n@@@@@@@@@@@@@@@@@@ NVI-DR: Results @@@@@@@@@@@@@@@@@@")
    print("@@@ J = {}, norme = {:.15e}, iterations = {}".format([round(x,10) for x in J],norme,k))
    print("@@@ Optimal Policy = ",Optimal_Policy)

    stats = pstats.Stats(profiler)
    ProcessTime = stats.total_tt
    print("@@@ Processing time NVI-DR algorithm = {} (s) ".format(ProcessTime))

def natural_Value_Iteration_Matricielle(P, Ria, max_iter, epsilon, discount):
    # Algorithme de Value Iteration pour Discounted  Reward
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ Natural Value Iteration NVI-DR : Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    profiler = cProfile.Profile() #For processing time
    profiler.enable()             #Démarrez le profilage

    # Initialisation de la valeur moyenne pour chaque état
    #J = [0 for _ in range(num_states)]
    J = np.zeros(N)
    Optimal_Policy = [0 for _ in range(N)]
    
    k = 0
    #print("---> Iteration k = ",k)
    #print("J = ",J)

    while(k <= max_iter):
        k += 1
        #print("---> Iteration k = ",k)
        J_prev = np.copy(J)

        #Calule de la nouvelle Q valeur, sur chaque etat 
        for s in range(N):
            Q = Ria[s, :] + discount*np.sum(P[:, s, :] * J_prev, axis=1)
            #print("Q = ",Q)

            a_opt = np.argmax(Q)
            Optimal_Policy[s] = a_opt
            J[s] = Q[a_opt]

        diff = J - J_prev
        norme = np.linalg.norm(diff, np.inf)
        #print("J = {}, norme = {:.15e} ".format([round(x,10) for x in J],norme))
        #print("Optimal Policy = ",Optimal_Policy)
        #print("J = {}, norme = {:.15e}, iteration = {}".format([round(x,10) for x in J],norme,k)) #rau = 0 in the natural VI, however we can obtain it (see Abhijit works)
        precision = epsilon*0.5*((1-discount)/discount) #0.000125 if epsilon = 1e-3 and discount = 0.8  (see Abhijit book)
        if norme < precision:
            break

    print("\n@@@@@@@@@@@@@@@@@@ NVI-DR: Results @@@@@@@@@@@@@@@@@@")
    print("@@@ J = {}, norme = {:.15e}, iterations = {}".format([round(x,10) for x in J],norme,k))
    print("@@@ Optimal Policy = ",Optimal_Policy)

    stats = pstats.Stats(profiler)
    ProcessTime = stats.total_tt
    print("@@@ Processing time for NVI-DR Matrixe algorithm = {} (s) ".format(ProcessTime))
    return Optimal_Policy

def natural_Value_Iteration_Matricielle_GS(P, Ria, max_iter, epsilon, discount):
    # Algorithme de Gauss Sedel Value Iteration pour Discounted  Reward
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ Natural Value Iteration NVI-GS-DR : Matrix GS version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    profiler = cProfile.Profile() #For processing time
    profiler.enable()             #Démarrez le profilage

    # Initialisation de la valeur moyenne pour chaque état
    #J = [0 for _ in range(num_states)]
    J = np.zeros(N)
    Optimal_Policy = [0 for _ in range(N)]
    
    k = 0
    #print("---> Iteration k = ",k)
    #print("J = ",J)

    while(k <= max_iter):
        k += 1
        #print("---> Iteration k = ",k)
        J_prev = np.copy(J)

        #Calule de la nouvelle valeur, sur chaque etat 
        for s in range(N):
            best = 1e-20

            for a in range(A):
                some = Ria[s, a] + discount*np.sum(P[a, s, :] * J) #updating "J" asynchronously !

                if(some > best): #recherche de la meilleure action
                    best = some
                    Optimal_Policy[s] = a
                    J[s] = best

        diff = J - J_prev
        norme = np.linalg.norm(diff, np.inf)
        #print("J = {}, norme = {:.15e} ".format([round(x,10) for x in J],norme))
        #print("Optimal Policy = ",Optimal_Policy)
        #print("J = {}, norme = {:.15e}, iteration = {}".format([round(x,10) for x in J],norme,k)) #rau = 0 in the natural VI, however we can obtain it (see Abhijit works)

        precision = epsilon*0.5*((1-discount)/discount) #0.000125 if epsilon = 1e-3 and discount = 0.8  (see Abhijit book)
        if norme < precision:
            break

    print("\n@@@@@@@@@@@@@@@@ NVI-GS-DR : Results @@@@@@@@@@@@@@@@@@@@")
    print("@@@ J = {}, norme = {:.15e}, iterations = {}".format([round(x,10) for x in J],norme,k)) #rau = 0 in the natural VI, however we can obtain it (see Abhijit works)
    print("@@@ Optimal Policy = ",Optimal_Policy)

    stats = pstats.Stats(profiler)
    ProcessTime = stats.total_tt
    print("@@@ Processing time for NVI-GS-DR matrixe algorithm = {} (s) ".format(ProcessTime))
    return Optimal_Policy

def relative_Value_Iteration_Matricielle(P, Ria, max_iter, epsilon, discount):
    # Algorithme de Relative Value Iteration pour Average Reward
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ Relative Value Iteration RVI-DR: Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    profiler = cProfile.Profile() #For processing time
    profiler.enable()             #Démarrez le profilage

    # Initialisation de la valeur moyenne pour chaque état
    J = [0 for _ in range(N)]
    Optimal_Policy = [0 for _ in range(N)]
    i = 0 #specific to RVI algorithm : 'i' selected randomly between 0 and num_states-1
    
    k = 0
    #print("---> Iteration k = ",k)
    #print("J = ",J)

    while(k <= max_iter):
        k += 1
        #print("---> Iteration k = ",k)
        J_prev = np.copy(J)

        #Calule de la nouvelle Q valeur, sur chaque etat 
        for s in range(N):
            Q = Ria[s, :] + discount*np.sum(P[:, s, :]*(J_prev), axis=1)
            #print("Q = ",Q)
            Q -= J_prev[0] #difference with Average, we don't retreive after "s" loop !

            a_opt = np.argmax(Q)
            Optimal_Policy[s] = a_opt
            J[s] = Q[a_opt]

        diff = J - J_prev
        norme = np.linalg.norm(diff, np.inf)
        #print("J = {}, norme = {:.15e} ".format([round(x,10) for x in J],norme))
        #print("Optimal Policy = ",Optimal_Policy)
        #print("J = {}, norme = {:.15e}, iteration = {}".format([round(x,10) for x in J],norme,k)) #rau = 0 in the natural VI, however we can obtain it (see Abhijit works)
        precision = epsilon*0.5*((1-discount)/discount) #0.000125 if epsilon = 1e-3 and discount = 0.8  (see Abhijit book)
        if norme < precision:
            break

    print("\n@@@@@@@@@@@@@@@@@@ RVI-DR: Results @@@@@@@@@@@@@@@@@@")
    print("@@@ J = {}, norme = {:.15e}, iterations = {}".format([round(x,10) for x in J],norme,k))
    print("@@@ Optimal Policy = ",Optimal_Policy)

    stats = pstats.Stats(profiler)
    ProcessTime = stats.total_tt
    print("@@@ Processing time for RVI-DR Matrixe algorithm = {} (s) ".format(ProcessTime))
    return Optimal_Policy

def policy_Evaluation(P, Ria, max_iter, epsilon, discount, policy):    #Policy evaluation iteratively
    print("@@@@ Policy evaluation : Matrix version @@@@@")
    print("policy = ",policy)

    H = np.zeros(N)
    k = 0
    #print("---> Iteration k = ",k)
    i = 0 #specific to RVI algorithm : 'i' selected randomly between 0 and num_states-1

    while(k <= max_iter):
        k += 1
        #print("---> Iteration k = ",k)
        H_prev = np.copy(H)

        #Calule de la nouvelle Q valeur, sur chaque etat 
        #Q = Ria[ :, policy ] + np.sum(P[policy, :, :] * Q_prev, axis=1)
        #print("H = ",H)
        for s in range(N):
            """print("P[policy[s], s, :] = ", P[policy[s], s, :])
            print("H_prev = ", H_prev)
            print("P[policy[s], s, :] * Q_prev = ",P[policy[s], s, :] * H_prev)
            print("np.sum(P[policy[s], s, :] * Q_prev) = ",np.sum(P[policy[s], s, :] * H_prev))
            print("Ria[s, policy[s]] = ",Ria[s, policy[s]])"""
            H[s] = Ria[s, policy[s]] + discount*np.sum(P[policy[s], s, :] * H_prev)
            #print("H[s] = ",H[s])
        #print ("H = ",H)

        diff = H - H_prev
        norme = np.linalg.norm(diff, np.inf)
        #print("H = {}, norme = {:.15e} ".format([round(x,10) for x in H],norme))
        #print("Optimal Policy = ",Optimal_Policy)
        
        precision = epsilon*0.5*((1-discount)/discount) #0.000125 if epsilon = 1e-3 and discount = 0.8  (see Abhijit book)  
        if norme < precision:
            break

    print ("H = {}, iterations = {}".format(H,k))
    return H

def relative_policy_Evaluation(P, Ria, max_iter, epsilon, discount, policy):    #Policy evaluation iteratively
    print("@@@@ Policy evaluation : Matrix version @@@@@")
    print("policy = ",policy)

    H = np.zeros(N)
    k = 0
    #print("---> Iteration k = ",k)
    i = 0 #specific to RVI algorithm : 'i' selected randomly between 0 and num_states-1

    while(k <= max_iter):
        k += 1
        #print("---> Iteration k = ",k)
        H_prev = np.copy(H)

        #Calule de la nouvelle Q valeur, sur chaque etat 
        #Q = Ria[ :, policy ] + np.sum(P[policy, :, :] * Q_prev, axis=1)
        #print("H = ",H)
        for s in range(N):
            """print("P[policy[s], s, :] = ", P[policy[s], s, :])
            print("H_prev = ", H_prev)
            print("P[policy[s], s, :] * Q_prev = ",P[policy[s], s, :] * H_prev)
            print("np.sum(P[policy[s], s, :] * Q_prev) = ",np.sum(P[policy[s], s, :] * H_prev))
            print("Ria[s, policy[s]] = ",Ria[s, policy[s]])"""
            H[s] = Ria[s, policy[s]] + discount*np.sum(P[policy[s], s, :] * H_prev)
            H[s] -= H_prev[i]

            #print("H[s] = ",H[s])

        diff = H - H_prev
        norme = np.linalg.norm(diff, np.inf)
        #span = np.max(diff) - np.min(diff)
        #print("H = {}, norme = {:.15e} ".format([round(x,10) for x in H],norme))
        #print("Optimal Policy = ",Optimal_Policy)
        
        precision = epsilon*0.5*((1-discount)/discount) #0.000125 if epsilon = 1e-3 and discount = 0.8  (see Abhijit book)  
        if norme < precision:
            break

    print ("H = {}, iterations = {}".format(H,k))
    return H

def policy_Evaluation_ROB(P, Ria, max_iter, epsilon, discount, policy): #Not optimale
    print("@@@@ Policy evaluation with ROB2 : Matrix version @@@@@")
    print("policy = ",policy)

    #Generation of TPM for policy "policy"
    matrix_policy = []
    R = []
    for s in range(N):
        matrix_policy.append(list(P[policy[s],s,:]))
        R.append(Ria[s, policy[s]])

    # Obtention de la matrice transposée
    matrix_policy = discount*np.array(matrix_policy)

    #print("matrixe = ",matrix_policy)
    #matrix_policy = matrix_policy.T
    #print("matrixe = ",matrix_policy)

    V = np.zeros(N)

    for p in range(N):
        if (matrix_policy[p,0] == 1): #States in "C"
            V[p] += R[p]

    for p in range(N-1,0,-1):
        if (matrix_policy[p,0] != 1): #Other states
            for i in range(p+1,N):
                V[p] += V[i]*matrix_policy[p,i]
            V[p] += R[p]

    print("V = ",V)

    return V

def policy_Improvement(P, Ria, max_iter, epsilon, discount, policy, H):
    print("@@@@ Policy improvement : Matrix version @@@@@")

    new_policy = np.copy(policy)

    #Calule de la nouvelle Q valeur, sur chaque etat 
    for s in range(N):
        Q = Ria[s, :] + discount*np.sum(P[:, s, :] * H, axis=1)
        #print("Q = ",Q)

        a_opt = np.argmax(Q)
        new_policy[s] = a_opt

    return new_policy

def policy_Iteration_Matricielle(P, Ria, max_iter, epsilon, discount):
    # Algorithme de "Modified Policy Iteration" pour Discounted  Reward
    # This algorithm uses "value iteration algorithm" in its evaluation step to 
    # avoid solving linear equation (in that step) which is time consuming for larger systems
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@ Policy Iteration PI-DR : Matrix version @@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    profiler = cProfile.Profile() #For processing time
    profiler.enable()             #Démarrez le profilage

    k = 0
    policy = [0 for _ in range(N)]
    #policy = [1,1]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= max_iter):
        k += 1
        print("---> Iteration k = {}".format(k))
        H = policy_Evaluation(P, Ria, max_iter, epsilon, discount, policy)
        #print("Policy = {}, H = {}, Average reward = {}".format(policy, H, rau))
        new_policy = policy_Improvement(P, Ria, max_iter, epsilon, discount, policy, H)

        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    print("\n@@@@@@@@@@@@@@@@ PI-DR : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)
    
    stats = pstats.Stats(profiler)
    ProcessTime = stats.total_tt
    print("@@@ Processing time for PI-DR Matrixe algorithm = {} (s) ".format(ProcessTime))
    
    return policy

def policy_Iteration_Matricielle_Relative(P, Ria, max_iter, epsilon, discount):
    # Algorithme de "Modified Policy Iteration" pour Discounted Reward
    # This algorithm uses "value iteration algorithm" in its evaluation step to 
    # avoid solving linear equation (in that step) which is time consuming for larger systems
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@ Policy Iteration Relative: Matrix version @@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    profiler = cProfile.Profile() #For processing time
    profiler.enable()             #Démarrez le profilage

    k = 0
    policy = [0 for _ in range(N)]
    #policy = [1,1]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= max_iter):
        k += 1
        print("---> Iteration k = {}".format(k))
        H = relative_policy_Evaluation(P, Ria, max_iter, epsilon, discount, policy)
        #print("Policy = {}, H = {}, Average reward = {}".format(policy, H, rau))
        new_policy = policy_Improvement(P, Ria, max_iter, epsilon, discount, policy, H)

        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    print("\n@@@@@@@@@@@@@@@@ PI-MV : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)
    
    stats = pstats.Stats(profiler)
    ProcessTime = stats.total_tt
    print("@@@ Processing time for PI-MV algorithm = {} (s) ".format(ProcessTime))
    return policy

def policy_Iteration_Matricielle_ROB(P, Ria, max_iter, epsilon, discount): #Not optimale
    # Algorithme de "Modified Policy Iteration" pour Discounted rewaed
    # This algorithm uses "value iteration algorithm" in its evaluation step to 
    # avoid solving linear equation (in that step) which is time consuming for larger systems
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@ Policy Iteration ROB : Matrix version @@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    profiler = cProfile.Profile() #For processing time
    profiler.enable()             #Démarrez le profilage

    k = 0
    policy = [0 for _ in range(N)]
    #policy = [1,1]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= max_iter):
        k += 1
        print("---> Iteration k = {}".format(k))
        
        H = policy_Evaluation_ROB(P, Ria, max_iter, epsilon, discount, policy)
        #print("Policy = {}, H = {}, Average reward = {}".format(policy, H, rau))
        new_policy = policy_Improvement(P, Ria, max_iter, epsilon, discount, policy, H)

        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    print("\n@@@@@@@@@@@@@@@@ PI-MV : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)
    
    stats = pstats.Stats(profiler)
    ProcessTime = stats.total_tt
    print("@@@ Processing time for PI-MV algorithm = {} (s) ".format(ProcessTime))
    return policy


#natural_Value_Iteration_Detailed(P, Ria, max_iter, epsilon, discount)
#natural_Value_Iteration_Matricielle(P, Ria, max_iter, epsilon, discount)
#policy_Iteration_Matricielle(P, Ria, max_iter, epsilon, discount)

#---------- Testing Value iterations --------------
"""policie1 = natural_Value_Iteration_Matricielle(P, Ria, max_iter, epsilon, discount)
policie2 = relative_Value_Iteration_Matricielle(P, Ria, max_iter, epsilon, discount)
policie3 = natural_Value_Iteration_Matricielle_GS(P, Ria, max_iter, epsilon, discount)
print("policie 1 = ",policie1)
print("policie 2 = ",policie2)
print("policie 3 = ",policie3)"""

"""t , c, tmax = 0, 0, 100
while(t< tmax):
    #Ria, P = generate_Model()
    policie1 = natural_Value_Iteration_Matricielle(P, Ria, max_iter, epsilon, discount)
    policie2 = relative_Value_Iteration_Matricielle(P, Ria, max_iter, epsilon, discount)
    #policie3 = natural_Value_Iteration_Matricielle_GS(P, Ria, max_iter, epsilon, discount)
    print("policie 1 = ",policie1)
    print("policie 2 = ",policie2)
    #print("policie 3 = ",policie3)
    if(all(a == b  for a, b in zip(policie1, policie2))):
        c+=1
    t += 1
print("Similarité = {}%".format((c/tmax)*100))"""

#---------- Testing Policy iterations --------------
"""policie1 = policy_Iteration_Matricielle(P, Ria, max_iter, epsilon, discount)
policie2 = policy_Iteration_Matricielle_Relative(P, Ria, max_iter, epsilon, discount)
#policie3 = natural_Value_Iteration_Matricielle_GS(P, Ria, max_iter, epsilon, discount)
print("policie 1 = ",policie1)
print("policie 2 = ",policie2)
#print("policie 3 = ",policie3)"""

"""t , c, tmax = 0, 0, 100
while(t< tmax):
    Ria, P = generate_Model()
    policie1 = policy_Iteration_Matricielle(P, Ria, max_iter, epsilon, discount)
    policie2 = policy_Iteration_Matricielle_Relative(P, Ria, max_iter, epsilon, discount)
    policie3 = policy_Iteration_Matricielle_ROB(P, Ria, max_iter, epsilon, discount)
    print("policie 1 = ",policie1)
    print("policie 2 = ",policie2)
    print("policie 3 = ",policie3)
    if(all(a == b and b==c  for a, b, c in zip(policie1, policie2, policie3))):
        c+=1
    t += 1
print("Similarité = {}%".format((c/tmax)*100))"""

