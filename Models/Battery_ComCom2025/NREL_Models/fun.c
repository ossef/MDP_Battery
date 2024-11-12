
/*---------------------------------------------*/
/*                                             */
/*partie de Code specifique a  chaque probleme */
/*                                             */
/*---------------------------------------------*/

/* Modéle DTMC de remplissage de batterie avec des "Energy Packets" */	
/* Modéle avec une phase "PV-ON" et une phase "PV-OFF" */
/* DTMC Descritpion : (X, H, S) */
		
/* Encoding discret events */
/* Arrivals = {0,1,2}, Service={0,1}, Release={0,1}, Phase={0,1} */
/* index : (Arrival, Service, Release, Modulating) */
/* 1 : (0, 0, 0, 0) - no arrival, no service, no release, no phase change */
/* 2 : (0, 0, 0, 1) - no arrival, no service, no release, phase change */
/* 3 : (0, 0, 1, 0) - ...      ...      ...        ...       ...        */
/* 4 : (0, 0, 1, 1) - ...      ...      ...        ...       ...        */

/* 23 : (2, 1, 1, 0) - 2 arrivals, service, release, no phase change */
/* 24 : (2, 1, 1, 1) - 2 arrivals, service, release, phase change    */



/* proba des evenements : arrival, service, Release, PhaseChange */ 

int a[100]; //Support of arrivals example {0, 1, 2, .... num_paquets}, from NSRDB Data !
int *a_size;

int b[NbService] = {0, 1};
int c[NbRelease] = {0, 1};
int d[NbPhase] = {0, 1};

//Not depending on action
double pa[100];                                    //proba of arrivals, example {0.05, 0.25, 0.4, 0.3}, from NSRDB Data !
double pAlpha = 0.01;                              //Alpha probability : (Working   PV => Deficient PV), 1% - 2% a year
double pBeta  = 0.99;                              //Beta probaility   : (Deficient PV => Working   PV)

//depending on action (max 5 actions for this scenario)
//double pService[5] = {0.1, 0.1, 0.1, 0.1, 0.1};        //Bernouilli service, for each action
double pRelease[5] = {0.1, 0.3, 0.5, 0.7, 0.9};        //Bernouilli release of Battery, for each action

//pService and pRelease for current action
double ps[2];
double pr[2];

typedef struct event
{
  int a; //Number of EP arrival
  int b; //service
  int c; //Battery release
  int d; //phase change
  double prob; //Product of events probas
} Event;
//Event events[NbEvtsPossibles];
Event *events; 

int *num_packets;
int *num_hours;
int *Deadline;
long double **hour_packet; //store distribution of EP arrivals, for each hour
double *pService; //store disctribution of DP arrival, for each hour


int min(int a, int b){
  return a<b ? a : b ;
}

int max(int a, int b){
  return a<b ? b : a ;
}

void InitEvents(int idAction){
   int i, j, k, l;
   int compt = 0;

   pr[1] = pRelease[idAction]; pr[0] = 1 - pr[1]; // Release probability, for actual action
   printf("Action %d : pRelease = %f \n", idAction+1, pr[1]);

  int NbArrivals = a_size[0];
  printf("NbArrivals = %d \n",NbArrivals);
  events = malloc( (NbArrivals * NbService * NbRelease * NbPhase) * sizeof(Event));

   for (i = 0; i < NbArrivals; i++) {
      for (j = 0; j < NbService; j++) {
        for (k = 0; k < NbRelease; k++) {
          for (l = 0; l < NbPhase; l++) {
              events[compt].a = i; 
              events[compt].b = j; 
              events[compt].c = c[k];
              events[compt].d = d[l];
              events[compt].prob = (double)pr[k];
              compt++;
              }
            }
          }
        }
}

void InitEtendue(int DebutHeure, int Buffer, int idAction)
{
  Min[0] = 0;
  Max[0] = Buffer;
  Min[1] = DebutHeure;
  Max[1] = Deadline[0];
  Min[2] = 0;
  Max[2] = 1;
  InitEvents(idAction);
}

// Fonction pour extraire la partie "ville_mois" de la chaîne "ville_mois_action"
char* extract(const char *s) {
    static char s_filtred[100];  // On utilise un buffer statique pour stocker la chaîne extraite
    strncpy(s_filtred, s, sizeof(s_filtred) - 1);
    
    // On trouve la position du dernier underscore '_'
    char *last_underscore = strrchr(s_filtred, '_');
    if (last_underscore != NULL) {
        *last_underscore = '\0';  // On coupe la chaîne avant le dernier underscore
    }

    return s_filtred;
}

int ReadDistribs(int Buffer, int idAction, char city[100], char s[100]) // 's' contient "Fairbanks_1991_a1"
{

    /* -------------  Reading DP arrival from 'Extracts/' -----------------  */

    char origin[150] = "../NREL_Extracts/";  // Buffer pour stocker le chemin complet
    strcat(origin, city);  // Concaténation de "../NSRDB_Extracts/" avec le nom de la ville

    char *s_filtred = extract(s);  // Extraction de la partie "ville_année" de la chaîne 's'

    char name[200];  // Buffer pour le nom complet du fichier
    snprintf(name, sizeof(name), "%s/%s_filtred_Dists.data", origin, s_filtred);

    //FILE *file = fopen("../NSRDB_Extracts/Fairbanks_1991_filtred_Dists.data", "r");
    FILE *file = fopen(name, "r");
    if (file == NULL) {
        printf("Impossible d'ouvrir le fichier dans '../../NREL_Extracts' \n");
        exit(0) ;
    }

    a_size       = malloc(1 * sizeof(int));
    num_packets  = malloc(1 * sizeof(int));
    num_hours    = malloc(1 * sizeof(int));
    Deadline     = malloc(1 * sizeof(int));

    char line[1000];
    // Sauter les lignes inutiles jusqu'à la matrice des probabilités
    fgets(line, sizeof(line), file);

    int d, f, packet_size;
    
    // Lire les dimensions du fichier (nombre d'heures et nombre de paquets)
    fscanf(file, "%d %d %d %d", &d, &f, &num_packets[0], &packet_size);
    printf("---> Hours = {%d ... Deadline = %d} \n",d,f);
    printf("---> For each hour, we have {0 ... %d} EP arrivals \n\n",num_packets[0]-1);

    num_hours[0] = f-d+1;
    Deadline[0]  = f;
    a_size[0]    = num_packets[0]; //size of EP arrivals support

    // Allouer dynamiquement la matrice des probabilités et le tableau des probabilités d'interruption
    hour_packet = malloc(num_hours[0] * sizeof(long double *));
    for (int i = 0; i < num_hours[0]; i++) {
        hour_packet[i] = malloc(num_packets[0] * sizeof(long double));
    }

    int *heures = malloc(num_hours[0] * sizeof(int));  // Stocker les heures

    int heure, i = 0;

    // Sauter les lignes inutiles jusqu'à la matrice des probabilités
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'H') break;
    }

    // Lecture de la matrice des probabilités heure x paquets
    for (i = 0; i < num_hours[0]; i++) {
        fscanf(file, "%d", &heures[i]);
        for (int j = 0; j < num_packets[0]; j++) {
            fscanf(file, "%Lf", &hour_packet[i][j]);
        }
    }

    fclose(file);

    // Exemple d'affichage des données lues
    /*
    printf("Matrice heure x paquets:\n");
    for (int h = 0; h < num_hours[0]; h++) {
        printf("Heure %d : ", heures[h]);
        for (int p = 0; p < num_packets[0]; p++) {
            printf("%lf ", hour_packet[h][p]);
        }
        printf("\n");
    }*/

    /* -------------  Reading DP arrival from 'Extracts/' -----------------  */
    char filename[200] = "../NREL_Extracts/Service_Demand.data";
    FILE *file2 = fopen(filename, "r");
    if (file2 == NULL) {
        printf("Erreur: Impossible d'ouvrir le fichier '%s'.\n",filename);
        exit(0) ;
    }

    pService = malloc(num_hours[0] * sizeof(double)); 

    int start_hour, end_hour;
    fscanf(file, "%d %d", &start_hour, &end_hour);  // Lire la première ligne avec les heures de début et de fin

    int hour;
    double probability;
    i = 0;
  
    // Lire les heures et les probabilités ligne par ligne
    while (fscanf(file2, "%d %lf", &hour, &probability) == 2) {
        if(hour>= d && hour <= f)
        {
          pService[i] = probability;
          i++;
        }
    }

    fclose(file2);
    InitEtendue(d, Buffer, idAction);

    return (a_size[0] * NbService * NbRelease * NbPhase) ;
}

void EtatInitial(E)
int *E;
{
  /*donne a E la valeur de l'etat racine de l'arbre de generation*/
  E[0] = Min[0];   // X
  E[1] = Min[1];   // H
  E[2] = 1;        // M
}


double Probabilite(indexevt, E)
int indexevt;
int *E;
{
  Event e = events[indexevt-1];
  double prob = 0;

  //--- 1) The Release: already calculated
  double p = events[indexevt-1].prob;

  //--- 2) The Phase : PV-OFF
  if(E[2] == 1)
  {
    if(e.d == 1)
      prob = pAlpha*p;
    else
      prob = (1-pAlpha)*p;
  }

  //--- 2) The Phase : PV-OFF
  if(E[2] == 0)
  {
    if(e.d == 1)
      prob = pBeta*p;
    else
      prob = (1-pBeta)*p;
  }

  //---  3) The service of DP (Data Packet)
  if(e.b == 1)
    prob *= pService[E[1]-Min[1]];
  else 
    prob *= (1-pService[E[1]-Min[1]]);

  //--- 4) The arrival of EP (Energy Packet)
  prob *= hour_packet[E[1]-Min[1]][e.a];

  //printf("--------> Transition (%d, %d, %d) : (%d, serv = %d, release =  %d, alpha = %d) = %lf \n",E[0],E[1],E[2],e.a,e.b,e.c,e.d,prob);

  return prob;
}


void Equation(E, indexevt, F, R, seuil)
int *E;
int indexevt;
int *F, *R;
double seuil;
{
  /*ecriture de l'equation d'evolution, transformation de E en F grace a l'evenemnt indexevt, mesure de la recompense R sur cette transition*/
  int bool = 0;
  F[0] = E[0]; // X
  F[1] = E[1]; // H
  F[2] = E[2]; // M
  Event e = events[indexevt-1];

  /*------------ "PV-ON" transitions -------------*/
  if(E[2] == 1)
  {
    if(e.d == 0) //no phase changing
    {
      // -------- H ---------
      if(E[1] < Max[1] && (E[0] < seuil || e.c == 0))
        {if(E[0] + e.a > Min[0] || E[1] > Min[1] ) //à rajouter dans formules, pour atteindre la premiére arrivé >0
          F[1]++; }
      if(E[1] == Max[1] || (E[0] >= seuil && e.c == 1))
        F[1] = Min[1];

      // -------- X ---------
      if(E[1] < Max[1] && E[0]<seuil )
        F[0] = max(min(F[0]+e.a, Max[0])-e.b, Min[0]);
      if(E[1] == Max[1])
        F[0] = Min[0];
      if(E[1] < Max[1] && E[0]>= seuil && e.c == 1)
        F[0] = Min[0];
      if(E[1] < Max[1] && E[0]>= seuil && e.c == 0)
        F[0] = max(min(F[0]+e.a, Max[0])-e.b, Min[0]);
    }
    else //phase changing
    {
      // -------- M ---------
      if(E[1]>=Min[1] && E[1]<Max[1])
        {F[0]=E[0]; F[1]=E[1]+1; F[2] = 1-E[2];}
      if(E[1] == Max[1]) //Deadline acheived, go to (0,0,1)
        {F[0]=Min[0]; F[1]=Min[1]; F[2] = E[2];}
    }
  }

  /*------------ "PV-OFF" transitions -----------*/
  else
  {
    if(e.d == 0) //no phase changing
    {
      // -------- H ---------
      if(E[1] < Max[1] && (E[0] < seuil || e.c == 0) )
          F[1]++;
      if(E[1] == Max[1] || (E[0] >= seuil && e.c == 1 ) )
        F[1] = Min[1];

      // -------- X ---------
      if(E[1] < Max[1] && E[0]<seuil )
        F[0] = max(F[0]-e.b, Min[0]);
      if(E[1] == Max[1])
        F[0] = Min[0];
      if(E[1] < Max[1] && E[0]>= seuil && e.c == 1 )
        F[0] = Min[0];
      if(E[1] < Max[1] && E[0]>= seuil && e.c == 0 )
        F[0] = max(F[0]-e.b, Min[0]);
    }
    else //phase changing
    {
      // -------- M ---------
      if( E[1]>=Min[1] && E[1]<Max[1]) //phase change ok
         {F[0]=E[0]; F[1]=E[1]+1; F[2] = 1-E[2];}
      if(E[1] == Max[1]) //Deadline acheived, go to (0,0,0)
        {F[0]=Min[0]; F[1]=Min[1]; F[2] = E[2];}
    }

    if(E[0] == Min[0] && E[1] == Min[1]) 
    {
      if(e.d == 1)
        {F[0]=Min[0]; F[1]=Min[1]; F[2] = 1-E[2];}
      else 
        {F[0]=E[0]; F[1]=E[1]; F[2] = E[2];}
    }
  }

  /*if(E[0] == 13 && E[1] == 11 && E[2] == 1 && indexevt-1 == 17)
  {
    printf("--------> Transition (%d, %d, %d) --> (%d, %d, %d) : (%d, serv = %d, release =  %d, alpha = %d) \n",E[0],E[1],E[2],F[0],F[1],F[2],e.a,e.b,e.c,e.d);
  }*/

  //if(E[0] == 0 && E[1] == 7 && E[2] == 1)
  //{
  //printf("---> State In   : (%d, %d, %d)\n",E[0],E[1],E[2]);
  //printf("%d Transition : Arriv = %d, Serv = %d, Release = %d, PhaseChange = %d \n",indexevt,e.a,e.b,e.c,e.d);
  //printf("State Out   : (%d, %d, %d)\n",F[0],F[1],F[2]);
  //}
}

void FreeMemory()
{
    // Free events tab
    /*int nevents = a_size[0] * NbService * NbRelease * NbPhase;
    for (int i = 0; i < nevents; i++) {
        free(events[i]);
    }*/
    free(events);
  
    /* // Free hour_packet tab
    for (int i = 0; i < num_hours[0]; i++) {
        free(hour_packet[i]);
    } */
    free(hour_packet);
    free(pService);
    free(num_hours);
    free(Deadline);
    free(num_packets);
}
