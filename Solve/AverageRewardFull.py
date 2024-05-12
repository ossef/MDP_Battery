import time
import numpy as np

#---------------- Average Reward functions -------------#

def steady_State_ROB_B(matrice_transition, N): #Proposed method for steady state distribution

    alpha, Pi = np.zeros(N), np.zeros(N)
    alpha[0] = 1
    
    #--- Calculate "alpha" vector --- v2 : perf
    for q in range(1,N):
        alpha[q] = np.sum(alpha[:q] * matrice_transition[:q,q])/(1-matrice_transition[q, q])

    """ #--- v1 : detailled
    for q in range(1,N):
        for p in range(q):
            alpha[q] += alpha[p]*matrice_transition[p][q]
    """

    #--- Calculate "Pi" vector --- v2 : perf
    s1 = np.sum(alpha[1:])
    Pi[0] = 1 / (1 + s1)
    Pi[1:] = alpha[1:] * Pi[0]

    """ #--- v1 : detailled
    s1 = 0
    for p in range(1,N):
        s1 += alpha[p]

    Pi[0] = 1/(1 + s1) 
    for q in range(1,N):
        Pi[q] = alpha[q]*Pi[0]"""

    if np.abs(sum(Pi) - 1) > 1e-10 :
        raise("Somme proba 'steady_State_ROB' = ",sum(Pi))

    #print("Pi ROB",Pi, "somme = ",sum(Pi))
    return Pi

def steady_State_Power(matrice_transition, N): #Proposed method for steady state distribution, for both with or without phases
    pi = np.ones(N) / N
    it = 0
    # Effectuer des itérations jusqu'à convergence ou jusqu'au nombre maximal d'itérations
    while(it <= 1e5):
        it+=1
        pi_old = np.copy(pi)
        pi = np.dot(pi, matrice_transition)

        # Vérifier la convergence
        norme = np.linalg.norm(pi - pi_old, ord=2)

        if norme < 1e-15:
            print("Iterations = ",it, " et nomre = ",norme)
            return pi

    raise("Méthode de puissance n'arrive pas à epsilon ")

def average_Reward_Power(matrice_transition, R, N): #Power method for steady state distribution
    pi = steady_State_Power(matrice_transition, N)
    rau = np.sum(pi*R)
    return rau

def average_Reward_ROB_B(matrice_transition,R, N):
    pi = steady_State_ROB_B(matrice_transition, N)
    rau = 0
    for s in range(N):
        rau += pi[s]*R[s]
    return rau


#---------------- Value Iteration Functions : Natural and Relative -------------#

def natural_Value_Iteration_Detailed(P, Ria, max_iter, epsilon, N, A):  # (NVI) Natural Value Iteration : Detailled version
    # Algorithme de Natural Value Iteration pour Average Reward
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@ Natural Value Iteration : Detailed version  @@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()
 
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
                Q[a] = Ria[s, a] + sum(P[a,s,s_next]*J_prev[s_next] for s_next in range(N))
            #print("Q = ",Q)

            tmp = list(Q.values())
            a_opt = np.argmax(tmp)
            Optimal_Policy[s]=a_opt
            J[s] = tmp[a_opt]

        diff = J - J_prev
        span = np.max(diff) - np.min(diff)
        #print("J = {}, span = {:.15e} ".format([round(x,10) for x in J],span))
        #print("Optimal Policy = ",Optimal_Policy)

        if span < epsilon:
            break

    #print("\n@@@@@@@@@@@@@@@@@@ NVI-DV: Results @@@@@@@@@@@@@@@@@@")
    #print("@@@ J = {}, span = {:.15e}".format([round(x,10) for x in J],span)) #rau = 0 in the natural VI, however we can obtain it (see Abhijit works)
    #print("@@@ Optimal Policy = ",Optimal_Policy)

    ProcessTime = time.time() - start_time
    print("@@@ Processing time NVI1 algorithm = {} (s) ".format(ProcessTime))

def natural_Value_Iteration_Matricielle(P, Ria, max_iter, epsilon, N, A):  # (NVI) Natural Value Iteration : Matricx version
    # Algorithme de Natural Value Iteration pour Average Reward
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ Natural Value Iteration: Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    # Initialisation de la valeur moyenne pour chaque état
    #J = [0 for _ in range(num_states)]
    J = np.zeros(N)
    Optimal_Policy = np.zeros(N)
    
    k = 0
    #print("---> Iteration k = ",k)
    #print("J = ",J)

    while(k <= max_iter):
        k += 1
        #print("---> Iteration k = ",k)
        J_prev = np.copy(J)

        #Calule de la nouvelle Q valeur, sur chaque etat 
        for s in range(N):
            Q = Ria[s, :] + np.sum(P[:, s, :] * J_prev, axis=1)
            #print("Q = ",Q)
            
            a_opt = np.argmax(Q)
            Optimal_Policy[s] = a_opt
            J[s] = Q[a_opt]

        diff = J - J_prev
        span = np.max(diff) - np.min(diff)
        #print("J = {}, span = {:.15e} ".format([round(x,10) for x in J],span))
        #print("Optimal Policy = ",Optimal_Policy)

        if span < epsilon:
            break

    #print("\n@@@@@@@@@@@@@@@@ NVI-MV : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ J = {}, span = {:.15e}, iterations = {}".format([round(x,10) for x in J],span,k)) #rau = 0 in the natural VI, however we can obtain it (see Abhijit works)
    #print("@@@ Optimal Policy = ",Optimal_Policy)

    #1) -----Generation of TPM for policy "policy"
    matrix_policy = np.empty((N, N))  # Initialisation d'un tableau vide pour matrix_policy
    R = np.empty(N)  # Initialisation d'un tableau vide pour R
    for s in range(N):
        matrix_policy[s] = P[int(Optimal_Policy[s]), s, :]
        R[s] = Ria[s, int(Optimal_Policy[s])].astype(float) 
    #2) -----Compute "rau" from steady-state
    rau = average_Reward_Power(matrix_policy,R)
    
    print("Iterations = ",k, " span = ",span)
    print("rau = {}".format(rau))
    ProcessTime = time.time() - start_time
    print("@@@ Processing time for NVI2 algorithm = {} (s) ".format(ProcessTime))
    return rau, Optimal_Policy

def relative_Value_Iteration_Detailed(P, Ria, max_iter, epsilon, N, A):  # (RVI) Relative Value Iteration : Detailled version
    # Algorithme de Relative Value Iteration pour Average Reward
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@ Relative Value Iteration  : Detailed version @@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()


    # Initialisation de la valeur moyenne pour chaque état
    #J = [0 for _ in range(N)]
    J = np.zeros(N)
    Optimal_Policy = np.zeros(N)
    i = 0 #specific to RVI algorithm : 'i' selected randomly between 0 and num_states-1

    k = 0
    #print("J = ",J)

    while(k <= max_iter):
        k += 1
        print("---> Iteration k = ",k)
        J_prev = np.copy(J)

        #Calule de la nouvelle Q valeur, sur chaque etat 
        for s in range(N):
            Q = {}
            for a in range(A):
                Q[a] = Ria[s, a] + sum(P[a,s,s_next]*J_prev[s_next] for s_next in range(N))
            #print("Q = ",Q)

            tmp = list(Q.values())
            a_opt = np.argmax(tmp)
            Optimal_Policy[s]=a_opt
            J[s] = tmp[a_opt]

        rau = J[i]                      #specific to RVI algorithm
        for s in range(N):     #specific to RVI algorithm
            J[s] -= rau                 #specific to RVI algorithm

        diff = J - J_prev
        span = np.max(diff) - np.min(diff)
        #print("J = {}, span = {:.15e}, rau = {:.5} ".format([round(x,10) for x in J],span,rau))
        #print("Optimal Policy = ",Optimal_Policy)

        if span < epsilon:
            break

    print("\n@@@@@@@@@@@@@@@@@@ RVI1 : Results @@@@@@@@@@@@@@@@@@")
    print("@@@ J = {}, span = {:.15e}, Average Reward = {:.5} ".format([round(x,10) for x in J],span,rau))
    print("@@@ Optimal Policy = ",Optimal_Policy)

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for RVI1  algorithm = {} (s) ".format(ProcessTime))

def relative_Value_Iteration_Matricielle(P, Ria, max_iter, epsilon, N, A):  # (RVI) Relative Value Iteration : Matrix version
    # Algorithme de Relative Value Iteration pour Average Reward
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ Relative Value Iteration: Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    # Initialisation de la valeur moyenne pour chaque état
    J = np.zeros(N)
    Optimal_Policy = np.zeros(N)
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
            Q = Ria[s, :] + np.sum(P[:, s, :] * J_prev, axis=1)
            #print("Q = ",Q)
            
            a_opt = np.argmax(Q)
            Optimal_Policy[s] = a_opt
            J[s] = Q[a_opt]

        rau = J[i]                     #specific to RVI algorithm
        J -= rau             #specific to RVI algorithm

        diff = J - J_prev
        span = np.max(diff) - np.min(diff)
        #print("J = {}, span = {:.15e}, rau = {:.5} ".format([round(x,10) for x in J],span,rau))
        #print("Optimal Policy = ",Optimal_Policy)

        if span < epsilon:
            break

    #print("\n@@@@@@@@@@@@@@@@ RVI-MV : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ J = {}, span = {:.15e}, rau = {:.5}, iterations = {} ".format([round(x,10) for x in J],span,rau,k))
    #print("@@@ Optimal Policy = ",Optimal_Policy)
    print("Iterations = ",k, " span = ",span)
    print("rau = {}".format(rau))
    ProcessTime = time.time() - start_time
    print("@@@ Processing time for RVI2 algorithm = {} (s) ".format(ProcessTime))
    return rau, Optimal_Policy


#---------------- Relative Policy Iteration Functions : GJ, FP, and ROB versions -------------#

def policy_Evaluation_FP(P, Ria, max_iter, epsilon, policy, N):  # Policy evaluation phase (FP): Fixed Point approx method
    print("@@@@ Relative Policy evaluation : Matrix version @@@@@")
    #print("policy = ",policy)

    H = np.zeros(N)
    k = 0
    #print("---> Iteration k = ",k)
    i = 0 #specific to RVI algorithm : 'i' selected randomly between 0 and num_states-1

    while(k <= max_iter):
        k += 1
        #print("---> Iteration k = ",k)
        H_prev = np.copy(H)

        #Calule de la nouvelle H valeur, sur chaque etat 
        for s in range(N):
            H[s] = Ria[s, policy[s]] + np.sum(P[policy[s], s, :] * H_prev)

        #print ("H = ",H)
        rau = H[i]
        H -= rau

        diff = H - H_prev
        span = np.max(diff) - np.min(diff)
        #print("J = {}, span = {:.15e} ".format([round(x,10) for x in J],span))
        #print("Optimal Policy = ",Optimal_Policy)

        if span < epsilon:
            break
    print("Iterations = ",k, " span = ",span)
    #print ("H = ",H)
    return rau, H

def policy_Evaluation_GJ(P, Ria, max_iter, epsilon, policy, N) : # Policy evaluation phase (GJ) : with Gauss Jordan elimination (not stable)
    # The linear equations to be solved are Gx=0. with G = I - P
    print("@@@@ Policy evaluation with Gauss-Jordan  @@@@@")
    #print("policy = ",policy)

    # I - Initializing a part of the G Matrix.
    G = np.zeros((N, N + 1))
    for row in range(N):
        for col in range(N) :
            if col == 0 :
                G[row][col]=1 #because the first value is replaced by rho 
            else :
                if(row==col) :
                    G[row][col]=1-P[policy[row]][row][col]
                else :
                    G[row][col]=-P[policy[row]][row][col]

     # Initializing the (NS+1)th column of G matrix 
    for state in range(N):
        G[state, N] = Ria[state, policy[state]] * np.sum(P[policy[state], state, :])

    # II - Gauss Jordan solver of GX = 0
    x = np.zeros(N)
    for col in range(N):
        # Trouver le meilleur pivot
        pivot = -0.1
        pivot_row = -1
        for row in range(col, N):
            if abs(G[row, col]) > pivot:
                pivot = abs(G[row, col])
                pivot_row = row

        # Vérifier si la solution peut être trouvée
        if pivot <= 1e-5:
            raise("Erreur dans l'évaluation de la politique GJ. Matrice singuliere.")

        # Échanger les lignes pour utiliser le meilleur pivot
        if pivot_row != col:
            G[[col, pivot_row]] = G[[pivot_row, col]]

        # Effectuer l'élimination
        for row1 in range(N):
            if row1 != col:
                factor = G[row1, col] / G[col, col]
                G[row1, col:] -= factor * G[col, col:]

    # Trouver la solution
    for row in range(N):
        x[row] = G[row, N] / G[row, row]

    rau = x[0]
    x[0] = 0 #the first value is set to 0
    #print("X = ",x)
    return rau, x #la moyenne et vecteur de valeurs

def policy_Evaluation_ROB_B(P, Ria, max_iter, epsilon, policy, N): # Proposed Policy evaluation (RB): exacte, direct and stable method 
    print("@@@@ Policy evaluation with ROB : Matrix version @@@@@")
    #print("policy = ",policy)

    #1) -----Generation of TPM for policy "policy"
    matrix_policy = np.empty((N, N))  # Initialisation d'un tableau vide pour matrix_policy
    R = np.empty(N)  # Initialisation d'un tableau vide pour R
    for s in range(N):
        matrix_policy[s] = P[policy[s], s, :]
        R[s] = Ria[s, policy[s]]
        
    #2) -----Compute "rau" from steady-state
    rau = average_Reward_ROB_B(matrix_policy,R,N)
    R -= rau

    #3) -----Initialize all values to "0"
    V = np.zeros(N)
    
    #3) -----Compute values of states in "C"
    C_states = matrix_policy[:, 0] == 1
    V[C_states] += R[C_states]

    #4) -----Compute values of other states "C"
    non_C_states = ~C_states  # Inverse de C_states
    for p in range(N-1, 0, -1):
        if non_C_states[p]:
            V[p] += np.sum(V[p+1:] * matrix_policy[p, p+1:])
            V[p] = (V[p] + R[p])/(1-matrix_policy[p,p])
            #V[p] += R[p]

    #print("V = ",V)
    return rau, V

def policy_Improvement(P, Ria, max_iter, epsilon, policy, H, N, A): # Policy improvement phase
    print("@@@@ Relative Policy improvement : Matrix version @@@@@")

    new_policy = np.copy(policy)

    #Calule de la nouvelle Q valeur, sur chaque etat 
    for s in range(N):
        Q = Ria[s, :] + np.sum(P[:, s, :] * H, axis=1)
        new_policy[s] = np.argmax(Q)
        #print("Q = ",Q)

    return new_policy

def policy_Iteration_Matricielle_FP(P, Ria, max_iter, epsilon, N, A):  # (RPI + FP) Relative Policy Iteration, using Fixed point approx 
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@         RPI + FP: Matrix version   @@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    k = 0
    policy = [0 for _ in range(N)]
    #policy = [1,1]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= max_iter):
        k += 1
        #print("---> Iteration k = {}".format(k))
        rau, H = policy_Evaluation_FP(P, Ria, max_iter, epsilon, policy, N)
        print("rau = {}".format(rau))
        #print("Policy = {}, H = {}, Average reward = {}".format(policy, H, rau))
        new_policy = policy_Improvement(P, Ria, max_iter, epsilon, policy, H, N, A)

        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-MV : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)
    #print("Average reward = {:.5}".format(rau))
    print("Iterations = ",k)
    ProcessTime = time.time() - start_time
    print("@@@ Processing time for RPI+FP algorithm = {} (s) ".format(ProcessTime))
    return rau, policy

def policy_Iteration_Matricielle_GJ(P, Ria, max_iter, epsilon, N, A): # (RPI + GJ) Relative Policy Iteration, using the Gauss-Jordan method
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@ RPI +  GJ : Matrix version            @@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    k = 0
    policy = [0 for _ in range(N)]
    #policy = [1,1]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= max_iter):
        k += 1
        #print("---> Iteration k = {}".format(k))
        rau, H = policy_Evaluation_GJ(P, Ria, max_iter, epsilon, policy, N)
        print("rau = {}".format(rau))
        #print("Policy = {}, H = {}, Average reward = {}".format(policy, H, rau))
        new_policy = policy_Improvement(P, Ria, max_iter, epsilon, policy, H, N, A)

        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-GJ-MV : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)
    #print("Average reward = {:.5}".format(rau))
    print("Iterations = ",k)
    ProcessTime = time.time() - start_time
    print("@@@ Processing time for RPI + GJ algorithm = {} (s) ".format(ProcessTime))
    return rau, policy

def policy_Iteration_Matricielle_ROB_B(P, Ria, max_iter, epsilon, N, A): # (RPI + RB) Relative Policy Iteration, using the Rob_B structure
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@ RPI +  ROB : Matrix version         @@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    k = 0
    policy = [0 for _ in range(N)]
    #policy = [1,1]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    while(k <= max_iter):
        k += 1
        #print("---> Iteration k = {}".format(k))
        
        rau, H = policy_Evaluation_ROB_B(P, Ria, max_iter, epsilon, policy, N) #you can teste with "policy_Evaluation_POWER""
        print("rau = {}".format(rau))
        #print("Policy = {}, H = {}, Average reward = {}".format(policy, H, rau))
        new_policy = policy_Improvement(P, Ria, max_iter, epsilon, policy, H, N, A)

        # Vérifier la convergence de la politique
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-MV : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)
    #print("Average reward = {:.5}".format(rau))
    print("Iterations = ",k)
    ProcessTime = time.time() - start_time
    print("@@@ Processing time for PI+ROB algorithm = {} (s) ".format(ProcessTime))

    return rau, policy

