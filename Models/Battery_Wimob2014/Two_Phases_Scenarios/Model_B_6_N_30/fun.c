
/*---------------------------------------------*/
/*                                             */
/*partie de Code specifique a  chaque probleme */
/*                                             */
/*---------------------------------------------*/

/* Modéle DTMC de remplissage de batterie avec des "Energy Packets" */	
/* Modéle avec une phase "Day" et une phase "Night" */
/* DTMC Descitpion : (X, H, S) */
		
/* Encoding discret events */
/* Arrivals = {0,1,2}, Service={0,1}, Release={0,1} */
/* index : (Arrival, Service, Release, Modulating) */
/* 1 : (0, 0, 0, 0) - no arrival, no service, no release, no phase change */
/* 2 : (0, 0, 0, 1) - no arrival, no service, no release, phase change */
/* 3 : (0, 0, 1, 0) - ...      ...      ...        ...       ...        */
/* 4 : (0, 0, 1, 1) - ...      ...      ...        ...       ...        */

/* 23 : (2, 1, 1, 0) - 2 arrivals, service, release, no phase change */
/* 24 : (2, 1, 1, 1) - 2 arrivals, service, release, phase change    */



/* proba des evenements : arrival, service, Release, PhaseChange */ 

int a[NbArrivals] = {0, 1, 2, 3}; 
int b[NbService] = {0, 1};
int c[NbRelease] = {0, 1};
int d[NbPhase] = {0, 1};

//Not depending on action
double pa[NbArrivals] = {0.05, 0.25, 0.4, 0.3};        //proba of arrivals, avg 1.3 arrival
double pAlpha = 1.0/720;                            //Alpha probability (D => N)
double pBeta  = 1.0/720;                           //Beta probaility (N => D)

//depending on action (max 5 actions for this scenario)
double pService[5] = {0.1, 0.1, 0.1, 0.1, 0.1};        //Bernouilli service, for each action
double pRelease[5] = {0.1, 0.3, 0.5, 0.7, 0.9};       //Bernouilli release of Battery, for each action

//pService and pRelease for current action
double ps[2];
double pr[2];

int min(int a, int b){
  return a<b ? a : b ;
}

int max(int a, int b){
  return a<b ? b : a ;
}

typedef struct event
{
  int a; //arrival
  int b; //service
  int c; //Battery release
  int d; //phase change
  double prob; //Product of events probas
} Event;
Event events[NbEvtsPossibles];

void InitEvents(int idAction){
   int i, j, k, l;
   int compt = 0;

   ps[1] = pService[idAction]; ps[0] = 1 - ps[1]; // Service probability, for actual action
   pr[1] = pRelease[idAction]; pr[0] = 1 - pr[1]; // Release probability, for actual action
   printf("Action %d : pService = %f \n", idAction+1, ps[1]);
   printf("Action %d : pRelease = %f \n", idAction+1, pr[1]);

   for (i = 0; i < NbArrivals; i++) {
      for (j = 0; j < NbService; j++) {
        for (k = 0; k < NbRelease; k++) {
          for (l = 0; l < NbPhase; l++) {
              events[compt].a = a[i]; 
              events[compt].b = b[j]; 
              events[compt].c = c[k];
              events[compt].d = d[l];
              events[compt].prob = (double)pa[i]*ps[j]*pr[k];
              compt++;
              }
            }
          }
        }
}

void InitEtendue(int Deadline, int Buffer, int idAction)
{
  Min[0] = 0;
  Max[0] = Buffer;
  Min[1] = 0;
  Max[1] = Deadline;
  Min[2] = 0;
  Max[2] = 1;
  InitEvents(idAction);
}


void EtatInitial(E)
int *E;
{
  /*donne a E la valeur de l'etat racine de l'arbre de generation*/
  E[0] = 0;
  E[1] = 0;
  E[2] = 1;
}


double Probabilite(indexevt, E)
int indexevt;
int *E;
{
  Event e = events[indexevt-1];
  double p = events[indexevt-1].prob;
  double prob = 0;

  //--- Day 
  if(E[2] == 1)
  {
    if(e.d == 1)
      prob = pAlpha*p;
    else
      prob = (1-pAlpha)*p;
  }

  //--- Night
  if(E[2] == 0)
  {
    if(e.d == 1)
      prob = pBeta*p;
    else
      prob = (1-pBeta)*p;
  }

  return prob;
}


void Equation(E, indexevt, F, R, seuil, Deadline)
int *E;
int indexevt,Deadline;
int *F, *R;
double seuil;
{
  /*ecriture de l'equation d'evolution, transformation de E en F grace a l'evenemnt indexevt, mesure de la recompense R sur cette transition*/
  int bool = 0;
  F[0] = E[0];
  F[1] = E[1];
  F[2] = E[2];
  Event e = events[indexevt-1];


  /*------------ Day transitions -------------*/
  if(E[2] == 1)
  {
    if(e.d == 0) //no phase changing
    {
      // -------- H ---------
      if(E[1] < Max[1] && (E[0] < seuil || e.c == 0))
        {if(E[0] + e.a > 0 || E[1] >0 ) //à rajouter dans formules, pour atteindre la premiére arrivé >0
          F[1]++; }
      if(E[1] == Max[1] || (E[0] >= seuil && e.c == 1))
        F[1] = 0;

      // -------- X ---------
      if(E[1] < Max[1] && E[0]<seuil )
        F[0] = max(min(F[0]+e.a, Max[0])-e.b, 0);
      if(E[1] == Max[1])
        F[0] = 0;
      if(E[1] < Max[1] && E[0]>= seuil && e.c == 1)
        F[0] = 0;
      if(E[1] < Max[1] && E[0]>= seuil && e.c == 0)
        F[0] = max(min(F[0]+e.a, Max[0])-e.b, 0);
    }
    else //phase changing
    {
      // -------- M ---------
      if(E[1]>0 && E[1]<Max[1])
        {F[0]=E[0]; F[1]=E[1]+1; F[2] = 1-E[2];}
      if(E[1] == Max[1]) //Deadline acheived, go to (0,0,1)
        {F[0]=Min[0]; F[1]=Min[1]; F[2] = E[2];}
    }
  }

  /*------------ Night transitions -----------*/
  else
  {
    if(e.d == 0) //no phase changing
    {
      // -------- H ---------
      if(E[1] < Max[1] && (E[0] < seuil || e.c == 0) )
          F[1]++;
      if(E[1] == Max[1] || (E[0] >= seuil && e.c == 1 ) )
        F[1] = 0;

      // -------- X ---------
      if(E[1] < Max[1] && E[0]<seuil )
        F[0] = max(F[0]-e.b, 0);
      if(E[1] == Max[1])
        F[0] = 0;
      if(E[1] < Max[1] && E[0]>= seuil && e.c == 1 )
        F[0] = 0;
      if(E[1] < Max[1] && E[0]>= seuil && e.c == 0 )
        F[0] = max(F[0]-e.b, 0);
    }
    else //phase changing
    {
      // -------- M ---------
      if( E[1]>0 && E[1]<Max[1]) //phase change ok
         {F[0]=E[0]; F[1]=E[1]+1; F[2] = 1-E[2];}
      if(E[1] == Max[1]) //Deadline acheived, go to (0,0,0)
        {F[0]=Min[0]; F[1]=Min[1]; F[2] = E[2];}
    }

    if(E[0] == 0 && E[1] == 0) 
    {
      if(e.d == 1)
        {F[0]=Min[0]; F[1]=Min[1]; F[2] = 1-E[2];}
      else 
        {F[0]=E[0]; F[1]=E[1]; F[2] = E[2];}
    }
  }

  /*if(F[0] == 8 && F[1] == 5 && F[2] == 0)
  {
    printf("Transition (%d, %d, %d) --> (%d, %d, %d) : (%d, %d, %d, alpha = %d) \n",E[0],E[1],E[2],F[0],F[1],F[2],e.a,e.b,e.c,e.d);
  }*/
}

void InitParticuliere()
{}
