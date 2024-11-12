import time
import numpy as np
from Graph import Graph
from scipy.sparse import csr_matrix, vstack, diags

discount = 0.8                   # in [0, 1] for futur rewards, only used in "DiscountedReward" cases 
epsilon = 1e-10                  # span epsilon precision
max_iter = 1e5
MDP_Case = 8   #Choose one of the models : The four last models are generated with "XBorne Tool, ISCIS 2016"
               #O : Not Robertazzi :  toy model of Gosavi book, page 147.
               #1 : Not Robertazzi :  large random models with random every-thing (probas, rewards, graph actions)
               #2 : Robertazzi     :  toy  RoberTazzi models (with the same graph, but different probas and rewars)
               #3 : Robertazzi     :  large Robertazzi models, with random every-thing (probas, rewards, graph actions)
               #4 : Robertazzi     :  large Ngreen Models, a model for optical container filling, details of the model in papaer "YHJM18" in "WIMOB 2018"
               #5 : Robertazzi     :  large Battery Models, only "Day" phase
               #6 : Robertazzi     :  large Battery Models, two phases "Day" and "night", scaling comparisons, results for "WIMOB 2024"
               #7 : Robertazzi     :  large Battery Models, two phases "Day" and "night", specific  scenario !, results for "WIMOB 2024"
               #8 : Robertazzi     :  Real NREL Data, two phases "Operating PV" and "Damaged PV", results for "ComCom 2025"

#--------- Not Robertazzi : A "Gosavi" book example, full matrix model : of N = 2 states and A = 2 actions
if MDP_Case == 0 : 
    N = 2
    A = 2

    def generate_MDP():
        # Matrices de transition P de taille : "2x2" pour chaque action. Au totale "ax2x2"
        P1 = [[0.7, 0.3],  #Matrice 1
              [0.4, 0.6]]
        P2 = [[0.9, 0.1],  #Matrice 2
              [0.2, 0.8]]
        All_P = np.array([P1, P2])  
        All_Ps = np.array([csr_matrix(P1), csr_matrix(P2)])  # A sparse version

        # Matrices de reward Riaj de taille : "2x2", une reward à partir d'un etat i vers j pour chaque action. Au totale "2x2xa"
        Riaj = np.array([[[6, -5], 
                        [7, 12]],       #Matrice 1
                        [[10, 17],      #Matrice 2
                        [-14, 13]]
                        ])  

        # Matrices de reward Ria de taille : "2xa", une reward à partir d'un etat i et une action
        Ria = np.array([[2.7, 10.7],   #etat1 : a1, a2 : 0.7*6+ 0.3*(-5), 0.9*10+ 0.1*17 
                        [10 , 7.6 ]    #etat2 : a1, a2 : 0.4*7 + 0.6*12, 0.2*(-14) + 0.8*13
                        ])
        return Ria, All_P , All_Ps, None

#--------- Not Robertazzi: A Large Full matrix random models : N and A to specify, for any matrixes
elif MDP_Case == 1 : 
    N = 100 #N=10000, A=10, AVG : BGS policy 1340s et Fixed 260s (eps=1e-15, Iter=1000)
    A = 10

    def generate_MDP():
        # Generate a random transition probability matrix for each action
        All_P = np.random.rand(A, N, N)
        # Normalize each row of the transition matrices to ensure they sum to 1
        All_P /= All_P.sum(axis=2, keepdims=True)

        # Generate a random reward matrix with integers
        Ria = np.random.randint(1, 100, size=(N, A))
        return Ria, All_P , None, None

#---------> RoberTazzi type "B" model : A Toy Full, in "Rob90" book example : of random probabilities and randow rewards
elif MDP_Case == 2 :
    N = 6
    A = 2

    def generate_MDP():
        x1, x2  = np.random.rand(), np.random.rand()
        t1, t2 = np.random.uniform(0.1, 0.5), np.random.uniform(0.1, 0.5)
        v1, v2 = np.random.uniform(0.1, 0.5), np.random.uniform(0.1, 0.5)

        Ria = np.random.randint(1, 100, size=(N, A))
        P1 = np.array([[0.1, 0.15, 0.2, 0.2, 0.35, 0 ],
                       [0, 0  , 0 , 1-v1-v2, v2, v1 ],
                       [0, 0  , 0 , 1-v1-v2, v1 ,v2 ],
                       [0, 0  , 0,  0 ,1-x1 ,x1 ],
                       [1, 0  , 0  , 0   , 0 , 0 ], 
                       [1, 0  , 0  , 0   , 0 , 0 ]
                ])

        P2 = np.array([[0.1, 0.2, 0.3, 0.2, 0.2, 0 ],
                       [0, 0  , 0 , 1-t1-t2, t2, t1 ],
                       [0, 0  , 0 , 1-t1-t2, t1 ,t2 ],
                       [0, 0  , 0,  0 ,1-x2 ,x2 ],
                       [1, 0  , 0  , 0   , 0 , 0 ], 
                       [1, 0  , 0  , 0   , 0 , 0 ]
                ])

        #All_P = np.array([P1, P2])
        All_Ps = np.array([csr_matrix(P1), csr_matrix(P2)])  # A sparse version
        All_P = None
        return Ria, All_P , All_Ps, None

#---------> RoberTazzi type "B" model : A Large "Sparse !" : of random graphes, probabilities and randow rewards
elif MDP_Case == 3 :
    #Model will be generated here, specify the following
    N = 1000 #Number of state
    A = 10   #Number of actions

    def generate_Matrix_Full():
        # Création d'une matrice P vide
        P = np.zeros((N, N))

        # Définir P[0,0] = 0, P[n-1,0] = 1, P[n-1,*] = 0 (où * signifie toutes les autres entrées de la dernière ligne)
        P[0, 0] = 0
        P[N-1, 0] = 1
        P[N-1, 1:] = 0

        # Générer des lignes avec le premier élément à 1 et le reste à 0
        for i in range(0, N-1):
            if np.random.rand() < 0.5 and i != 0:  # Probabilité arbitraire de choisir une telle ligne
                P[i, 0] = 1
            else:
                # Générer des valeurs aléatoires pour les éléments après l'indice i
                P[i, i+1:] = np.random.rand(N - i - 1)
                # Normalisation de la ligne
                P[i, i+1:] = P[i, i+1:] / P[i, i+1:].sum()
        return P,csr_matrix(P)
    
    def generate_Matrix_Sparse():
        data = []
        rows = []
        cols = []

        for i in range(N):
            if i < N - 1:
                if np.random.rand() < 0.5 and i != 0:
                    # Ligne avec le premier élément à 1 et le reste à 0
                    data.append(1)
                    rows.append(i)
                    cols.append(0)
                else:
                    # Générer des valeurs aléatoires uniquement pour les colonnes j+1, j+2, et j+3
                    max_col = min(N, i + 4)  # Assurer que nous ne dépassons pas la taille de la matrice
                    row_data = np.random.rand(max_col - i - 1)
                    row_data /= row_data.sum()  # Normalisation de la ligne

                    data.extend(row_data)
                    rows.extend([i] * len(row_data))
                    cols.extend(range(i + 1, max_col))
            else:
                # Dernière ligne : P[N-1, 0] = 1, P[N-1, 1:] = 0
                data.append(1)
                rows.append(N - 1)
                cols.append(0)

        P = csr_matrix((data, (rows, cols)), shape=(N, N))
        #print(P.toarray())
        return None, P

    def generate_MDP():
        #P = []
        ALL_Ps = []
        #generate a random Robertazzi matrix for each action
        for _ in range(A):
            p, ps = generate_Matrix_Sparse()
            #P.append(p)
            ALL_Ps.append(ps)

        #P = np.array(P)
        #Ps = np.array(Ps)
        Ria = np.random.randint(1,100, size=(N, A))

        #print("=>  Creating Full Matrixes from sparse for testing ")
        All_P = None
        return Ria, All_P , ALL_Ps, None

#---------> RoberTazzi type "B" NGreen optical model, "Sparse, Ngreen, Xborne" : data from "./Models/Ngreen_Wimob2018/" directory
elif MDP_Case == 4:
    #Generated with XBorne tool, stored in "./Models/Ngreen_Wimob2018/" directory

    index_model = 2 #choose one of the models
    models = [50, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 100000]
    sizes  = [52, 501, 1006, 2010, 5050, 10028, 20078, 30009, 50001, 100015]

    model= '../Models/Ngreen_Wimob2018//Model_N_'+str(models[index_model])+'/'

    N = sizes[index_model]  
    A = 10  #Max = 100 actions 

    def generate_MDP() :
        print("=>  Reading all sparse matrixes")
        print("A = ", A," actions")
        start_time = time.time()

        myGraph = Graph(model, None, None, 0, "WiMob18")  #Initalize and read a ".sz" file, "0" pour le modéle sans phase
        N = myGraph.N 

        #--------- read TPM(a) : Matrixes model from external file for --------
        All_Ps = []
        for a in range(A) :
            myGraph.read_Rii_Matrixe(a, 0, "WiMob18")
            Ps = myGraph.csr_sparse
            All_Ps.append(Ps)
        All_Ps = np.array(All_Ps)

        #----------- creating random rewards ---------------------------------
        Ria = np.random.normal(loc=50, scale=15, size=(N, A))
        #print("Ria = ",Ria)

        ProcessTime = time.time() - start_time
        print("=>  Reading all sparse matrixes ... Done in : {} (s) ".format(ProcessTime))

        #All_P = [p.toarray() for p in All_Ps]
        #All_P = np.array(All_P)
        All_P = None

        print("=>  Creating Full Matrixes from sparse ")
        return Ria, All_P , All_Ps, None

#---------> RoberTazzi type "B" approx Battery model, one phase !, "Sparse, Battery, Xborne" : data from "./Models/Battery_Wimob2024/One_Phase" directory
elif MDP_Case == 5:
    #Generated with XBorne tool, stored in "./Models/Battery_Wimob2024/One_Phase" directory

    index_model = 0 #choose one of the models
    Buffer = [6, 8, 100, 500, 1000]
    models = [14, 20, 2000, 62000, 160000]
    sizes  = [14, 22, 2109, 62375, 160800]

    model= '../Models/Battery_Wimob2024/One_Phase/Model_B_'+str(Buffer[index_model])+'_N_'+str(models[index_model])+'/'

    N = sizes[index_model]  
    A = 10  #Max = 100 actions 

    def generate_MDP() :
        print("=>  Reading all sparse matrixes")
        print("A = ", A," actions")
        start_time = time.time()

        myGraph = Graph(model, None, None, 0, "WiMob24") 
        N = myGraph.N 

        #--------- read TPM(a) : Matrixes model from external file for --------
        All_Ps = []
        for a in range(A) :
            myGraph.read_Rii_Matrixe(a, 0, "WiMob24")
            Ps = myGraph.csr_sparse
            All_Ps.append(Ps)
        All_Ps = np.array(All_Ps)

        #----------- creating random rewards ---------------------------------
        Ria = np.random.normal(loc=50, scale=15, size=(N, A))
        #print("Ria = ",Ria)

        ProcessTime = time.time() - start_time
        print("=>  Reading all sparse matrixes ... Done in : {} (s) ".format(ProcessTime))

        #All_P = [p.toarray() for p in All_Ps]
        #All_P = np.array(All_P)
        All_P = None

        print("=>  Creating Full Matrixes from sparse ")
        return Ria, All_P , All_Ps, None

#---------> RoberTazzi type "B" model, Battery model, two phases !, "Sparse, Battery, Xborne" : data from "./Models/Battery_Wimob2024/Two_Phases_Scaling/" directory
elif MDP_Case == 6:
    #Generated with XBorne tool, stored in ./Models/Battery_Wimob2024/Two_Phases_Scaling/" directory

    index_model = 3 #choose one of the  models
    #          0    1    2   3     4    5       6      7      8      9      10      11,    12
    Buffer = [15,  18,  25,  32,  50,   80,    100,   130,   150,   320,   450,    700,    1450]
    models = [100, 200, 300, 500, 1000, 3000,  5000,  8000,  10000, 50000, 100000, 200000, 1000000]
    sizes  = [113, 219, 313, 545, 1013, 2965,  5101,  8065,  9941 , 50245 ,99905,  199081, 1001113] 

    model= '../Models/Battery_Wimob2024/Two_Phases_Scaling/Model_B_'+str(Buffer[index_model])+'_N_'+str(models[index_model])+'/'

    N = sizes[index_model]  
    A = 100  #MaxActions = 100 actions for index<= 11, index = 12 => MaxActions = 20

    def generate_MDP() :
        print("=>  Reading all sparse matrixes")
        print("A = ", A," actions")
        start_time = time.time()

        #myGraph = Graph(model,1)  
        myGraph = Graph(model, None, None, 1, "WiMob24")
        N = myGraph.N 

        #--------- read TPM(a) : Matrixes model from external file for --------
        All_Ps = []
        for a in range(A) :
            myGraph.read_Rii_Matrixe(a, 1, "WiMob24")
            Ps = myGraph.csr_sparse
            All_Ps.append(Ps)
        All_Ps = np.array(All_Ps)

        #----------- creating random rewards ---------------------------------
        Ria = np.random.normal(loc=50, scale=15, size=(N, A))
        #print("Ria = ",Ria)

        ProcessTime = time.time() - start_time
        print("=>  Reading all sparse matrixes ... Done in : {} (s) ".format(ProcessTime))

        #All_P = [p.toarray() for p in All_Ps]
        #All_P = np.array(All_P)
        All_P = None

        return Ria, All_P , All_Ps, None

#---------> RoberTazzi type "B" model, Battery model, two phases !, "Sparse, Battery, Xborne" : data from "./Models/Battery_Wimob2024/Two_Phases_Scenarios/" directory
elif MDP_Case == 7:
    #Generated with XBorne tool, stored in "./Models/Battery_Wimob2024/Two_Phases_Scenarios/" directory

    #            0   1   2 
    Buffer    = [6, 20, 40]
    models    = [30, 215, 450]
    sizes     = [31, 215, 457] 
    Seuils    = [1, 5, 20]
    Deadlines = [3, 7, 12]
    Arrivals  = [0,1,3,5] #Batch of arrivals
    pArrivals = [0.05, 0.25, 0.4, 0.3]
    pAlpha = 1.0/720
    pBeta  = 1.0/720 
    pService = [0.9, 0.1] #no-service, serivce

    index_model = 1 #choose one of the models

    model= '../Models/Battery_Wimob2024/Two_Phases_Scenarios/Model_B_'+str(Buffer[index_model])+'_N_'+str(models[index_model])+'/'
    BufferSize  = Buffer[index_model]        #Capacité de la batterie
    seuil       = Seuils[index_model]
    N           = sizes[index_model]         #Number of states
    A           = 5                          #Number of actions (needs to be as maximum as possible as in directory "model")

    def generate_Reward(ALL_Ps, states, r1, r2, r3, r4):
        # Initialiser la matrice de récompense
        reward_matrix = np.zeros((N, A))
        '''r1 = 30     #Reward, for battery release
        r2 = -30      #Penalty empty battery (i.e. packets delay)
        r3 = -100     #Penalty of packets losts
        r4 = -1      #Penalty Loop at (0,0,1)'''
        #r5 = 2
        for id1, etat1 in enumerate(states):
            x1, t1, m1 = etat1[0], etat1[1], etat1[2]
            for a in range(A):
                s = 0
                for id2, etat2 in enumerate(states):
                    r = 0
                    x2, t2, m2 = etat2[0], etat2[1], etat2[2]
                    if(ALL_Ps[a][id1,id2] >  0 and id1>0 and id1<N-1):
                        if (t2 == 0) :    #Reward, for battery release
                            r = x1*r1
                            s += ALL_Ps[a][id1,id2]*r
                            #print('==> action {}, (x2={}, t2={}, m2={}) : r1+ = {}'.format(a+1,x2,t2,m2,r))
                        if(x2 == BufferSize):                     #Penalty of packets losts
                            r = 0
                            for e in range(len(Arrivals)):
                                if m1 == 1 : #Day
                                    module = pAlpha if m2 == 0 else 1 - pAlpha
                                if m1 == 0 : #Night
                                    module = pBeta  if m2 == 0 else 1 - pBeta
                                for b in range(2) :
                                    r += module*pArrivals[e]*pService[b]*max(0,x1+Arrivals[e]-b-BufferSize)*r2
                                    #r += max(0,x1+Arrivals[e]-b-BufferSize)*r2 #module*pArrivals[e]*pService[b]*max(0,x1+Arrivals[e]-b-BufferSize)*r2
                                    '''if (x1+Arrivals[e]-b-BufferSize > 0):
                                        print('{}, {}, {} --> {}, {}, {} : {}'.format(x1,t1,m1,x2,t2,m2, x1+Arrivals[e]-b-BufferSize ))'''
                            s+=r
                            #s += ALL_Ps[a][id1,id2]*r
                            #print('==> action {}, (x2={}, t2={}, m2={}) : r4- = {}'.format(a+1,x2,t2,m2,r))
                        if (x2 == 0 and t2>0 and x1>0 ) :                  #Penalty empty battery (i.e. packets delay)
                            r = r3
                            #s+=r
                            s += ALL_Ps[a][id1,id2]*r
                            #print('==> action {}, (x2={}, t2={}, m2={}) : r2- = {}'.format(a+1,x2,t2,m2,r))
                        if(id1 == 0 and id2 == 0) :               #Penalty Loop at (0,0,1)
                            r = pArrivals[0]*r4
                            s+=r
                            #s += ALL_Ps[a][id1,id2]*r
                            #print('==> action {}, (x2={}, t2={}, m2={}) : r3- = {}'.format(a+1,x2,t2,m2,r))
                        '''if(x2 > x1):
                            r = (x2-x1)*r5
                            s += ALL_Ps[a][id1,id2]*r
                            print('==> action {}, (x2={}, t2={}, m2={}) : r5+ = {}'.format(a+1,x2,t2,m2,r))'''
                reward_matrix[id1][a] = s

        #print("Ria = ",reward_matrix)
        return reward_matrix

    def generate_MDP(All_Ps, states, r1, r2, r3, r4, read) :
        if(read == 0):
            print("=>  Reading all sparse matrixes")
            print("A = ", A," actions")
            start_time = time.time()

            myGraph = Graph(model, None, None, 1, "WiMob24")  #Initalize and read a ".sz" file, '1' for model with two phases
            N = myGraph.N 

            #--------- read TPM(a) : Matrixes model from external file for --------
            All_Ps = []
            for a in range(A) :
                myGraph.read_Rii_Matrixe(a,1,"WiMob24")
                Ps = myGraph.csr_sparse
                All_Ps.append(Ps)
            All_Ps = np.array(All_Ps)
            states = myGraph.states
            
            ProcessTime = time.time() - start_time
            print("=>  Reading all sparse matrixes ... Done in : {} (s) ".format(ProcessTime))

        #----------- creating rewards ---------------------------------
        #Ria = np.random.normal(loc=50, scale=15, size=(N, A)) #Random rewards
        Ria = generate_Reward(All_Ps, states, r1, r2, r3, r4)          #Defined rewards

        All_P = None
        return Ria, All_P , All_Ps, states

#---------> RoberTazzi type "B" model, Battery model, two phases !, "Sparse, Battery, Xborne" : NSRDB Real data from "./Models/Battery_ComCom2025/" directory
elif MDP_Case == 8 :
    #Generated with XBorne tool, stored in "./Models/Battery_ComCom2025/" directory

    pAlpha = 0.01
    pBeta  = 0.99 
    pRelease = [0.1, 0.3, 0.5, 0.7, 0.9] #for each action

    A           = 5                          #Number of actions (needs to be as maximum as possible as in directory "model")

    def generate_Reward(N, Thr, BufferSize, h_deb, h_fin, packet_size, n_packets, hours_packets, ALL_Ps, states, r1, r2, r3, r4):
        # Initialiser la matrice de récompense
        reward_matrix = np.zeros((N, A))
        for id1, etat1 in enumerate(states):
            x1, t1, m1 = etat1[0], etat1[1], etat1[2]
            for a in range(A):
                s = 0
                for id2, etat2 in enumerate(states):
                    x2, t2, m2 = etat2[0], etat2[1], etat2[2]

                    #Reward, for battery release 
                    if (t2 == h_deb) :
                        s += ALL_Ps[a][id1,id2]*x1*r1
                        #if ALL_Ps[a][id1,id2]>0 and x1 == 0:
                        #    print("A",a+1," :  (x1, t1, m1) = ",x1, t1, m1," ---> (x2, t2, m2) = ",x2, t2, m2, "gagné +",ALL_Ps[a][id1,id2]*r, "prob = ",ALL_Ps[a][id1,id2])

                    #Penalty for packets loss
                    if(m1 == 1 and x2 == BufferSize):       
                        r = 0
                        for e in range(n_packets):
                            for b in range(2) :
                                #r += hours_packets[t1-h_deb][e]*pService[b]*max(0,x1+e-b-BufferSize)*r2
                                r += ALL_Ps[a][id1,id2]*max(0,x1+e-b-BufferSize)*r2
                        s+=r
                        #s += ALL_Ps[a][id1,id2]*r
                        #print('==> action {}, (x2={}, t2={}, m2={}) : r4- = {}'.format(a+1,x2,t2,m2,r))

                    #Penalty for empty battery
                    if (x2 == 0) : #and t2>h_deb and x1>0 ) :      #Penalty empty battery (i.e. packets delay)
                        s += ALL_Ps[a][id1,id2]*r3
                        #print('==> action {}, (x2={}, t2={}, m2={}) : r2- = {}'.format(a+1,x2,t2,m2,r))
                    '''
                    if(id1 == 0 and id2 == 0) :               #Penalty Loop at (0,0,1)
                        r = hours_packets[0][0]*r4
                        s+=r
                    '''
                        #s += ALL_Ps[a][id1,id2]*r
                        #print('==> action {}, (x2={}, t2={}, m2={}) : r3- = {}'.format(a+1,x2,t2,m2,r))
                    '''if(x2 > x1):
                        r = (x2-x1)*r5
                        s += ALL_Ps[a][id1,id2]*r
                        print('==> action {}, (x2={}, t2={}, m2={}) : r5+ = {}'.format(a+1,x2,t2,m2,r))'''
                reward_matrix[id1][a] = s

        #print("Ria = ",reward_matrix)
        return reward_matrix

    def read_Dists(nom_fichier):
        
        #-----------Reading EP arrivals distributions -------------#
        with open(nom_fichier, 'r') as f:
            # Sauter la première ligne qui est l'en-tête "Matrice des probabilités (heure x paquets):"
            f.readline()
            
            # Lire la deuxième ligne pour obtenir heure_debut, heure_fin, nombre_paquets, et paquet_size
            premiere_ligne = f.readline().strip().split()
            heure_debut    = int(premiere_ligne[0])
            heure_fin      = int(premiere_ligne[1])
            nombre_paquets = int(premiere_ligne[2])
            packet_size    = int(premiere_ligne[3])
            
            # Ignorer la ligne des en-têtes "Heure 0 1 2 3 4"
            f.readline()

            # Initialiser une matrice vide pour stocker les probabilités (heures x paquets)
            heures = np.arange(heure_debut, heure_fin + 1)
            matrice_probabilites = np.zeros((len(heures), nombre_paquets))

            # Lire les lignes suivantes et remplir la matrice
            for i, ligne in enumerate(f):
                valeurs = list(map(float, ligne.strip().split()[1:]))  # On ignore la première colonne (l'heure)
                matrice_probabilites[i, :] = valeurs

        #-----------Reading DP arrivals distribution  -------------#
        pService = {}  # Dictionnaire pour stocker les probabilités par heure
        filename = '../Models/Battery_ComCom2025/NREL_Extracts/Service_Demand.data'
        with open(filename, "r") as file:
            first_line = file.readline() #ignone first line
            for line in file:
                hour, probability = line.split()
                pService[int(hour)] = float(probability)  # Stocker la probabilité par heure

        return heure_debut, heure_fin, nombre_paquets, packet_size, matrice_probabilites, pService

    def generate_MDP(All_Ps, states, r1, r2, r3, r4, read, city, number, DATA_TYPE) :

        if(read == 0):
            print("=>  Reading all sparse matrixes")
            print("A = ", A," actions")
            start_time = time.time()

            if DATA_TYPE == 1 :
                model= '../Models/Battery_ComCom2025/NSRDB_Models/'+city+'/'
            if DATA_TYPE == 2 :
                model= '../Models/Battery_ComCom2025/NREL_Models/'+city+'/'

            myGraph = Graph(model, city, number, 1, "ComCom25")  #Initalize and read a ".sz" file, '1' for model with two phases
            N     = myGraph.N 
            Thr   = myGraph.Thr
            BufferSize = myGraph.Buffer

            #--------- read TPM(a) : Matrixes model from external file for --------
            All_Ps = []
            for a in range(A) :
                myGraph.read_Rii_Matrixe(a, 1, "ComCom25")
                Ps = myGraph.csr_sparse
                All_Ps.append(Ps)
            All_Ps = np.array(All_Ps)
            states = myGraph.states

            ProcessTime = time.time() - start_time
            print("=>  Reading all sparse matrixes ... Done in : {} (s) ".format(ProcessTime))

            #----------- Preparation for rewards ---------------------------------
            if DATA_TYPE == 1 : 
                filename = '../Models/Battery_ComCom2025/NSRDB_Extracts/'+city+'/'+city+'_'+number+'_filtred_Dists.data'
            if DATA_TYPE == 2 : 
                filename = '../Models/Battery_ComCom2025/NREL_Extracts/'+city+'/'+city+'_'+number+'_filtred_Dists.data'

            h_deb, h_fin, n_packets, packet_size, hours_packets, pService =  read_Dists(filename)

        #----------- Creating  ---------------------------------
        Ria = generate_Reward(N, Thr, BufferSize, h_deb, h_fin, packet_size, n_packets, hours_packets, All_Ps, states, r1, r2, r3, r4)          #Defined rewards

        All_P = None
        return N, n_packets, packet_size, hours_packets, pService, BufferSize, Thr, h_deb, h_fin, Ria, All_P , All_Ps, states