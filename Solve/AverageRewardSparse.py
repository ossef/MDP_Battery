import time
import numpy as np
from scipy.sparse import csr_matrix, vstack, diags

#---------------- Average Reward functions -------------#

def steady_State_ROB_A(matrice_transition, N): #Proposed method for steady state distribution 

    Pi = np.zeros(N)
    Pi[0] = 1
 
    for q in range(1, N):
        s = np.sum(matrice_transition[:q-1, q-1] * Pi[:q-1])
        Pi[q] = (Pi[q-1] * (1 - matrice_transition[q-1, q-1]) - s) / matrice_transition[q, q-1]

    """for q in range(1,N):
        s = 0
        for p in range(q-1):
            s += matrice_transition[p][q-1]*Pi[p]
        Pi[q] = (Pi[q-1]*(1-matrice_transition[q-1][q-1]) - s)/matrice_transition[q][q-1] #Boucles autorisés"""

    s = np.sum(Pi[1:])
    Pi[0] = 1 / (1 + s)
    Pi[1:] = Pi[1:] * Pi[0]

    print("Pi_ROB_A : ",Pi, "somme = ",sum(Pi))
    if np.abs(sum(Pi) - 1) > 1e-10 :
        raise("Erreur, somme proba 'steady_State_ROBA' = ",sum(Pi))

    return Pi

def steady_State_ROB_B(matrice_transition, N): #Proposed method for steady state distribution, for both with or without phases

    alpha, Pi = np.zeros(N), np.zeros(N)
    alpha[0] = 1

    #--- Calculate "alpha" vector --- v2 : perf
    for q in range(1, N):
        colonne_q = matrice_transition.getcol(q)
        alpha[q] = (alpha[:q].dot(colonne_q[:q].toarray().ravel()))/(1-matrice_transition[q, q])


    #--- Calculate "Pi" vector --- v2 : perf
    s1 = sum(alpha[1:])
    Pi[0] = 1 / (1 + s1)
    Pi[1:] = alpha[1:] * Pi[0]

    """ #--- v1 : detailled
    for q in range(1,N):
        for p in range(q):
            alpha[q] += alpha[p]*matrice_transition[p][q]

    s1 = 0
    for p in range(1,N):
        s1 += alpha[p]

    Pi[0] = 1/(1 + s1) 
    for q in range(1,N):
        Pi[q] = alpha[q]*Pi[0]"""

    if abs(sum(Pi) - 1) > 1e-10 :
        raise("Somme proba 'steady_State_ROB' = ",sum(Pi))

    #print("Pi_RobB : ",Pi, "somme = ",sum(Pi))
    return Pi

def steady_State_Power(matrice_transition,N): #Power method for steady state distribution
    it = 0
    pi = np.ones(N) / N
    if isinstance(matrice_transition, csr_matrix):
        matrice_transition = matrice_transition.toarray() # en plein
    # Effectuer des itérations jusqu'à convergence ou jusqu'au nombre maximal d'itérations

    while (it <= 1e5):
        it+=1
        pi_old = np.copy(pi)
        pi = np.dot(pi, matrice_transition)

        # Vérifier la convergence
        norme = np.linalg.norm(pi - pi_old, ord=2)

        if norme < 1e-15:
            print("Iterations = ",it, " et nomre = ",norme)
            return pi

    raise("Méthode de puissance n'arrive pas à epsilon")

def average_Reward_Power(matrice_transition,R,N):
    pi = steady_State_Power(matrice_transition,N)
    rau = np.dot(pi,R)
    return rau

def average_Reward_ROB_B(matrice_transition,R,N):
    pi = steady_State_ROB_B(matrice_transition,N)
    rau = np.dot(pi,R)
    return rau

def average_Measures(P, states, policy, N, packet_size, Buffer, Thr, h_deb, Deadline, pRelease, pService, n_packets, hours_packets):
    # 1) Generation of TPM for policy "policy"
    matrix_policy_rows = [P[policy[s]][s] for s in range(N)]
    matrix_policy = vstack(matrix_policy_rows)    # Fusionner les lignes en une matrice CSR
    pi = steady_State_ROB_B(matrix_policy,N)

    energy, lossRate, noService = 0, 0, 0

    for i, (x, h, m) in enumerate(states): 

        #---------- E[Release] ------------------------
        if h == Deadline:
            energy += x * pi[i]*packet_size
        if x>= Thr and h<Deadline :
            energy += x * pi[i] * pRelease[policy[i]]*packet_size

        #---------- E[lost_EP] ------------------------
        if x >= n_packets and m == 1 :
            for e in range(n_packets):
                for b in range(2) :
                    if(e-b>0):
                        if b == 1:
                            lossRate += hours_packets[h-h_deb][e] * pService[h] * pi[i] * max(0,x+e-b-Buffer)*packet_size
                        else : 
                            lossRate += hours_packets[h-h_deb][e] * (1-pService[h]) * pi[i] * max(0,x+e-b-Buffer)*packet_size

        #---------- Proba[noService] ------------------------
        if x == 0 :
            #print('states (x={}, h={}, m={}) : proba = {} '.format(x,h,m, pi[i]))
            noService += pi[i]*pService[h]

    return energy, lossRate, noService

#---------------- Value Iteration Functions : Natural and Relative -------------#

def natural_Value_Iteration_Csr(P, Ria, max_iter, epsilon, N, A) :  # (NVI) Natural Value Iteration : "Sparse CSR" version for matrixes   
    # Algorithme de Ntural Value Iteration pour Average Reward Sparse Row
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@ NVI Algorithm : Sparse Matrix version @@@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
    
    start_time = time.time()
    J = np.zeros(N)
    Optimal_Policy = np.zeros(N)

    k = 0
    while(k <= max_iter):
        k += 1
        J_prev = J.copy()

        Q = np.zeros((N, A))
        for a in range(A):
            Q[:, a] = Ria[:, a] + P[a].dot(J_prev)   #Q[:, a] = Ria[:, a] + P[a].dot(J_prev)
        J = np.max(Q, axis=1)

        """#Calule de la nouvelle Q valeur, sur chaque etat 
        for s in range(N):
            Q = [Ria[s, a] + P[a][s, :].dot(J_prev) for a in range(A)]
            a_opt = np.argmax(Q)
            Optimal_Policy[s] = a_opt
            J[s] = Q[a_opt][0]"""

        diff = J - J_prev
        span = max(diff) - min(diff)
        #print("Iteration k = {}, span = {:.15e}".format(k,span))
        #print("k = {}, J = {}, span = {:.15e} ".format(k,[round(x,10) for x in J],span))
        #print("Optimal Policy = ",Optimal_Policy)
        if span < epsilon :
            break


    Optimal_Policy = np.argmax(Q, axis=1)
    #print("\n@@@@@@@@@@@@@@@@ NVI-MV : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ J = {}, span = {:.15e}".format([round(x,10) for x in J],span)) #rau = 0 in the natural VI, however we can obtain it (see Abhijit works)
    #print("@@@ Optimal Policy = ",Optimal_Policy)

    print("Iteration k = {}, span = {:.15e}".format(k,span))
    # -----Compute "rau" from steady-state
    matrix_policy_rows = [P[Optimal_Policy[s]][s] for s in range(N)]
    matrix_policy = vstack(matrix_policy_rows)  # Fusionner les lignes en une matrice CSR
    R = Ria[np.arange(N), Optimal_Policy].astype(float)   # Vecteur des récompenses pour la politique donnée
    rau = average_Reward_ROB_B(matrix_policy,R,N)
    print("rau = ",rau)


    ProcessTime = time.time() - start_time
    print("@@@ Processing time for NVI algorithm = {} (s) ".format(ProcessTime))
    return rau, Optimal_Policy

def relative_Value_Iteration_Csr(P, Ria, max_iter, epsilon, N, A) :  # (RVI) Relative Value Iteration for "Sparse CSR" actions matrixes   
    # Algorithme de Relative  Value Iteration pour Average Reward Sparse Row
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@ RVI algorithm : Sparse Matrix version @@@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()

    # Initialisation de la valeur moyenne pour chaque état
    #J = [0 for _ in range(num_states)]

    J = np.zeros(N)
    Optimal_Policy = np.zeros(N)
    #historique_variations = []  # Pour suivre l'historique des variations
    i, k = 0, 0

    while(k <= max_iter):
        k += 1
        #J_prev = np.copy(J)
        J_prev = J.copy()

        Q = np.zeros((N, A))
        for a in range(A):
            Q[:, a] = Ria[:, a] + P[a].dot(J_prev)   #Q[:, a] = Ria[:, a] + P[a].dot(J_prev)
        J = np.max(Q, axis=1)

        rau = J[i]
        J[i] = 0
        J[i+1:] -= rau

        """Q = np.zeros((N, A))
        for a in range(A):
            Q[:, a] = Ria[:, a] + P[a].dot(J_prev)
        J = np.max(Q, axis=1)"""

        """#Calule de la nouvelle Q valeur, sur chaque etat 
        for s in range(N):
            Q = [Ria[s, a] + P[a][s, :].dot(J_prev) for a in range(A)]
            a_opt = np.argmax(Q)
            Optimal_Policy[s] = a_opt
            J[s] = Q[a_opt][0]"""

        diff = J - J_prev
        span = max(diff) - min(diff)

        # Vérifier la précision
        if span < epsilon :
            break

    Optimal_Policy = np.argmax(Q, axis=1)
    print("Iteration k = {}, rau = {}, span = {:.15e}".format(k,rau,span))
    #print("policy =",Optimal_Policy)

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for RVI algorithm = {} (s) ".format(ProcessTime))
    return rau, Optimal_Policy


#---------------- Relative Policy Iteration Functions : GJ, FP, and ROB versions -------------#

def policy_Evaluation_Csr_FP(P, Ria, max_iter, epsilon, policy,N): # Policy evaluation phase (FP): Fixed Point approx method
    print("@@@@ Relative Policy evaluation : Sparse  Matrix version @@@@@")
    
    H = np.zeros(N)
    
    k = 0
    #print("---> Iteration k = ",k)
    i = 0 #specific to RVI algorithm : 'i' selected randomly between 0 and num_states-1
    #historique_variations = []  # Pour suivre l'historique des variations

    P_preextracted = [P[policy[s]][s] for s in range(N)]
    while(k <= max_iter):
        k += 1
        #print("---> Iteration k = ",k)
        H_prev = H.copy()

        for s in range(N):
            H[s] = Ria[s, policy[s]] +  P_preextracted[s].dot(H_prev)[0] 

        rau = H[i]
        H -= rau

        diff = H - H_prev
        span = max(diff) - min(diff)
        #historique_variations.append(span)
        #print("H = {}, span = {:.15e}, rau ".format([round(x,10) for x in H],span,rau))
        #print("Optimal Policy = ",Optimal_Policy)
        #print("Iteration k = {}, span = {:.15e}".format(k,span))
        # Verifier précision
        if span < epsilon:
            break

        """# Sinon, Vérifier la stabilisation
        if k >= fenetre_glissante:
            variation_moyenne = np.mean(historique_variations[-fenetre_glissante:])
            variation_relative = abs(variation_moyenne - span) / variation_moyenne

            if variation_relative < seuil_relative_stabilite:
                print(f"Stabilité atteinte après {k} itérations.")
                break

    # ----- Vérifier stabilité numérique de "rau"
    matrix_policy_rows = [P[policy[s]][s] for s in range(N)]
    matrix_policy = vstack(matrix_policy_rows)  # Fusionner les lignes en une matrice CSR
    R = Ria[np.arange(N), policy].astype(float)   # Vecteur des récompenses pour la politique donnée
    rauRob = average_Reward_ROB_B(matrix_policy,R)
    close = np.ptp(np.array([rau,rauRob])) <= 1e-8
    if close == False :
        print("diff = ",abs(rau-rauRob))
        rau = rauRob"""

    print("Iteration k = {}, span = {:.15e}".format(k,span))
    return rau, H

def policy_Evaluation_Csr_ROB_B(P, Ria, max_iter, epsilon, policy, N): # Proposed Policy evaluation (RB): exacte, direct and stable method 
    #print("@@@@ Policy evaluation with ROB : Sparse Matrix version @@@@@")

    #1) -----Generation of TPM for policy "policy"
    """matrix_policy = []  # Initialisation d'un tableau vide pour matrix_policy
    R = np.empty(N)  # Initialisation d'un tableau vide pour R
    for s in range(N):
        matrix_policy.append(P[policy[s]][s])
        R[s] = Ria[s, policy[s]]"""

    # 1) Generation of TPM for policy "policy"
    matrix_policy_rows = [P[policy[s]][s] for s in range(N)]
    matrix_policy = vstack(matrix_policy_rows)  # Fusionner les lignes en une matrice CSR
    R = Ria[np.arange(N), policy].astype(float)   # Vecteur des récompenses pour la politique donnée

    #2) -----Compute "rau" from steady-state
    rau = average_Reward_ROB_B(matrix_policy,R,N)
    R -= rau

    #3) -----Initialize all values to "0"
    V = np.zeros(N)
    
    # 3) Compute values of states in "C"
    C_states = matrix_policy.getcol(0).toarray().ravel() == 1
    V[C_states] += R[C_states]

    # 4) Compute values of other states of "C"
    non_C_states = ~C_states  # Inverse de C_states
    for p in range(N-1, 0, -1):
        if non_C_states[p]:
            ligne_p =  matrix_policy_rows[p].toarray().ravel()
            V[p] += np.dot(V[p+1:], ligne_p[p+1:])
            #V[p] += np.sum(V[p+1:] * ligne_p[p+1:])
            V[p] = (V[p] + R[p])/(1-matrix_policy[p,p])

    #print("V = {}, rau = {}".format(V,rau))
    return rau, V

def policy_Evaluation_Csr_GJ(P, Ria, max_iter, epsilon, policy, N) : # Policy evaluation phase (GJ) : with Gauss Jordan elimination (not stable)
    # The linear equations to be solved are Gx=0. with G = I - P
    print("@@@@ Policy evaluation with Gauss-Jordan @@@@@")
    #print("policy = ",policy)

    # I - Initializing a part of the G Matrix.
    G = np.zeros((N, N + 1))
    for row in range(N):
        P_row_dense = P[policy[row]][row].toarray().ravel()  # Conversion de la ligne CSR en format dense
        for col in range(N) :
            if col == 0 :
                G[row][col] = 1  # because the first value is replaced by rho 
            else :
                if row == col :
                    G[row][col] = 1 - P_row_dense[col]
                else :
                    G[row][col] = -P_row_dense[col]

    # Initializing the (NS+1)th column of G matrix 
    for state in range(N):
        G[state, N] = Ria[state, policy[state]] * sum(P[policy[state]][state].toarray().ravel())


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

"""
def policy_Improvement_Csr(P, Ria, max_iter, epsilon, policy, H, N, A): # Policy improvement phase
    #print("@@@@ Relative Policy improvement : Sparse Matrix version @@@@@")

    new_policy = policy.copy()

    Q = np.zeros((N, A))
    for a in range(A):
        Q[:, a] = Ria[:, a] + P[a].dot(H)   
    new_policy= np.argmax(Q, axis=1)

    return new_policy
"""

def policy_Improvement_Csr(P, Ria, max_iter, epsilon, policy, H, N, A): 
    new_policy = policy.copy()

    Q = np.zeros((N, A))
    for a in range(A):
        Q[:, a] = np.round(Ria[:, a] + P[a].dot(H),10)  # Calcul de Q pour chaque action
    new_policy= np.argmax(Q, axis=1)

    """
    # Amélioration de la politique avec critère de sélection
    for s in range(N):
        max_q_value = np.max(Q[s, :])
        # Sélectionner les actions dont la valeur Q est proche du max, avec une tolérance epsilon
        close_actions = np.where(np.abs(Q[s, :] - max_q_value) <= 1e-10)[0]
        # Choisir l'action avec l'indice le plus faible parmi celles proches du max
        new_policy[s] = np.min(close_actions)
    """

    return new_policy

def policy_Iteration_Csr_FP(P, Ria, max_iter, epsilon, N, A): # (RPI + FP) Relative Policy Iteration, using Fixed point approx 
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ RPI + FP algorithm  : Sparse Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()
    history = []

    k = 0
    policy = [0 for _ in range(N)]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    rau = -1e15
    while(k <= max_iter):
        k += 1
        raup = rau
        #print("---> Iteration k = ",k)
        rau, H = policy_Evaluation_Csr_FP(P, Ria, max_iter, epsilon, policy, N)
        #print("Policy = {}, H = {}, Average reward = {}".format(policy, H, rau))
        print("rau = {}".format(rau))
        new_policy = policy_Improvement_Csr(P, Ria, max_iter, epsilon, policy, H, N, A)

        # Vérifier la convergence de la politique
        history.append(rau)
        if (rau < raup) or np.array_equal(new_policy, policy) or (len(history)>20 and np.ptp(history[-20]) <= 1e-10) or k>500:
            break


        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-MV : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)
    #print("Average reward = {:.5}".format(rau))
    print("Iterations = {}, rau = {}".format(k,rau))

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for RPI+FP algorithm = {} (s) ".format(ProcessTime))
    return rau, policy

def policy_Iteration_Csr_GJ(P, Ria, max_iter, epsilon, N, A): # (RPI + GJ) Relative Policy Iteration, using the Gauss-Jordan method
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@ RPI + GJ algorithm  : Sparse Matrix version @@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()
    history = []

    k = 0
    policy = [0 for _ in range(N)]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    rau = -1e15
    while(k <= max_iter):
        k += 1
        raup = rau
        #print("---> Iteration k = ",k)
        rau, H = policy_Evaluation_Csr_GJ(P, Ria, max_iter, epsilon, policy, N)
        #print("Policy = {}, H = {}, Average reward = {}".format(policy, H, rau))
        print("rau = {}".format(rau))
        new_policy = policy_Improvement_Csr(P, Ria, max_iter, epsilon, policy, H, N, A)

        # Vérifier la convergence de la politique
        history.append(rau)
        if (rau < raup) or np.array_equal(new_policy, policy) or (len(history)>20 and np.ptp(history[-20]) <= 1e-10) or k>500:
            break

        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-MV : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)
    #print("Average reward = {:.5}".format(rau))
    print("Iterations = {}, rau = {}".format(k,rau))

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for RPI+GJ algorithm = {} (s) ".format(ProcessTime))
    return rau, policy

def policy_Iteration_Csr_ROB_B(P, Ria, max_iter, epsilon, N, A): # (RPI + RB) Relative Policy Iteration, using the Rob_B structure
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@ RPI + RB algorithm  : Sparse Matrix version @@@@@")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    start_time = time.time()
    history = []

    k = 0
    policy = [0 for _ in range(N)]
    #print("---> Iteration k = {}, initial policy = {}".format(k,policy))

    #Etracting all first column states with 1 entry : only once
    #C_states = P[0].getcol(0).toarray().ravel() == 1
    rau = -1e15
    while(k <= max_iter):
        print("rau : ",rau)
        k += 1
        raup = rau
        #print("---> Iteration k = ",k)
        rau, H = policy_Evaluation_Csr_ROB_B(P, Ria, max_iter, epsilon, policy, N)
        #print("Policy = {}, H = {}, Average reward = {}".format(policy, H, rau))
        #print("rau = {}".format(rau))
        new_policy = policy_Improvement_Csr(P, Ria, max_iter, epsilon, policy, H, N, A)

        history.append(rau)
        if (rau < raup) or np.array_equal(new_policy, policy) or (len(history)>20 and np.ptp(history[-20]) <= 1e-10) or k>500:
            break

        policy = new_policy

    #print("\n@@@@@@@@@@@@@@@@ PI-MV : Results @@@@@@@@@@@@@@@@@@@@")
    #print("@@@ H = {}, Average reward = {:.5} ".format([round(x,10) for x in H],rau))
    #print("@@@ Optimal Policy = ",policy)
    #print("Average reward = {:.5}".format(rau))
    print("Iterations = {}, rau = {}".format(k,rau))

    ProcessTime = time.time() - start_time
    print("@@@ Processing time for RPI+RB algorithm = {} (s) ".format(ProcessTime))
    return rau, policy