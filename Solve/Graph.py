from scipy.sparse import csr_matrix

class Graph:
    def __init__(self, model, city, number, index, type):
        #Number could be a "year" number as "2024" or a "mounth" number as "M6"
        #fileSz = model+'Model_a1-reordre.sz' if index == 1 else  model+'Model_a1.sz'

        if(type == "ComCom25") :
            fileSz = model+city+'_'+number+'_a1-reordre.sz' if index == 1 else  model+city+'_'+number+'_a1.sz'
            with open(fileSz, 'r') as file:
                lines = file.readlines()
                l4 = lines[3].split()
                
                M, N, X, Thr, Buffer  = int(lines[0]), int(lines[1]), int(lines[2]), int(l4[0]), int(l4[1])
                print("(M, N, X, Thr, Buffer) = ", M, N, X, Thr, Buffer)

            fileCd = model+city+'_'+number+'_a1-reordre.cd' if index == 1  else  model+city+'_'+number+'_a1.cd'
            states = []
            with open(fileCd, 'r') as file:
                lines = file.readlines()
                if(index == 0) : #Modéle sans phase : (X, T)
                    for line in lines :
                        elemts = line.split()
                        id, x, t = int(elemts[0]), int(elemts[1]), int(elemts[2])
                        states.append((x,t))
                else :  #Modéle avec phase : (X, T, M)
                    for line in lines :
                        elemts = line.split()
                        id, x, t, m = int(elemts[0]), int(elemts[1]), int(elemts[2]), int(elemts[3])
                        states.append((x,t,m))

            #print("Etats : ",states)
            self.N      = N
            self.Thr    = Thr 
            self.Buffer = Buffer
            self.model  = model
            self.city   = city 
            self.number = number
            self.states = states
            self.csr_sparse = None

        if(type == "WiMob24" or type == "WiMob18") :
            fileSz = model+'Model_a1-reordre.sz' if index == 1 else  model+'Model_a1.sz'
            with open(fileSz, 'r') as file:
                lines = file.readlines()
                M, N, X = int(lines[0]), int(lines[1]), int(lines[2])
                print("(M, N, X) = ", M, N, X)

            fileCd = model+'Model_a1-reordre.cd' if index == 1  else  model+'Model_a1.cd'
            states = []
            with open(fileCd, 'r') as file:
                lines = file.readlines()
                if(index == 0) : #Modéle sans phase : (X, T)
                    for line in lines :
                        elemts = line.split()
                        id, x, t = int(elemts[0]), int(elemts[1]), int(elemts[2])
                        states.append((x,t))
                else :  #Modéle avec phase : (X, T, M)
                    for line in lines :
                        elemts = line.split()
                        id, x, t, m = int(elemts[0]), int(elemts[1]), int(elemts[2]), int(elemts[3])
                        states.append((x,t,m))

            #print("Etats : ",states)
            self.N = N
            self.model = model
            self.states = states
            self.csr_sparse = None

    def showLine(self,id):
        print("Line[{}] = {}",id,self.csr_sparse[id])

    def showGraph(self):
        print("Graph = ",self.csr_sparse)

    def read_Rii_Matrixe(self,action,index,type):

        if(type == "ComCom25") :
            fileRii =  self.model+self.city+'_'+self.number+'_a'+str(action+1)+'-reordre.Rii' if index == 1 else  self.model+self.city+'_'+self.number+'_a'+str(action+1)+'.Rii'
        if(type == "WiMob24" or type == "WiMob18") :
            fileRii = self.model+'Model_a'+str(action+1)+'-reordre.Rii' if index == 1 else  self.model+'Model_a'+str(action+1)+'.Rii'

        self.csr_sparse = None
        rows, cols, data = [], [], [] 
        with open(fileRii, 'r') as file:
            lines = file.readlines()
            for line in lines: 
                elemts = line.split()
                node, degre = int(elemts[0]), int(elemts[1])
                for d in range(1,degre+1) : 
                    proba, state = float(elemts[2*d]), int(elemts[2*d+1])
                    rows.append(node)
                    cols.append(state)
                    data.append(proba)
        self.csr_sparse = csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        #print("sparse = ",self.csr_sparse)
        #self.showGraph()
        #myGraph.showLine(6)
        #return myGraph
        #myGraph.showLine(6)