
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct liste_sommets
      {
	int 		som_global;
        double 		val;
        struct liste_sommets 	*suiv;
      };

struct matgth
      {
     	double 		somme; 
     	struct liste_sommets	*lig;
     	struct liste_sommets 	*col;
      };

FILE                   *G;
char 		*fic_resultat ;
struct matgth	       *mat;
double             *pi;
time_t                 tps, avant1_user, avant_user, apres1_user, apres_user, temps_total;
struct tm              *tpsecoule;
/* unsigned int         tpsmax,tt; */
struct liste_sommets   *llc, *ll;
int                    N, j;
double             S,norm;
char 		*pt_buffer_pi, *buffer_pi ; /* bufferisation des pi[i]
						avant ecriture fichier */

/* Cette procedure affiche le temps */

void  Affich_clock() 
{
 apres1_user = time(&apres_user);
 tps = apres1_user - avant1_user; 
 tpsecoule  = gmtime(&tps);
 
 printf("h: %d m: %d sec : %d\n", tpsecoule->tm_hour, tpsecoule->tm_min,tpsecoule ->tm_sec);
}

 

/* Insere un element colonne de la matrice. */

void Inserer_Colonne(int som,double val,int pred)

{
     struct liste_sommets   *lc, *l; 
     int                     fin;

    l  = NULL;

    lc = mat[som].col;
    if(mat[som].col == NULL)
      {
	if((mat[som].col= (struct liste_sommets *) 
                          calloc (1, sizeof(struct liste_sommets))) == NULL)
          exit(1);
	else
	  {
	    mat[som].col->som_global = pred;
            mat[som].col->val        = val;
            mat[som].col->suiv       = NULL;
	  }
      }
    else /* il y a au moins un element dans la colonne */ 
      {
       if (lc->som_global == pred)
          lc->val = lc->val + val;
       else /* lc->som_global != pred */
         {
           if(lc->som_global >  pred)
             {
	       if((l= ( struct liste_sommets *) 
                             calloc (1 , sizeof(struct liste_sommets)))==NULL) 
                { 
                  exit(1);
                }
	       else
	         {
                   l->som_global = pred;
                   l->val        = val;
		   l->suiv       = lc;	      
                   mat[som].col  = l;
	         }
              }
            else 
              {
	        fin =0;		
                while ((lc->suiv != NULL) && (fin==0)) 
	          {
		    if(lc->suiv->som_global  < pred)
		      lc = lc->suiv;
		    else 
                      fin = 1;
		  } 
                if(lc->suiv != NULL) 
		  {
		    if (lc->suiv->som_global == pred)
                       lc->suiv->val = lc->suiv->val +  val;
		    else 
                      {
                        if ((l = ( struct liste_sommets *) 
                             calloc (1, sizeof(struct liste_sommets))) == NULL)
			   exit(1);    
			else
			  { 
                            l->som_global = pred;
              		    l->val        = val;
              		    l->suiv       = lc->suiv;
			    lc->suiv       = l;
			  } 
                      }
		   }
                 else 
                   {                       
		     if((l = ( struct liste_sommets *) 
                         calloc (1, sizeof(struct liste_sommets))) == NULL)
		        exit(1);
		     else
		       {	  
                         l->som_global = pred;
                         l->val        = val;
		         l->suiv       = NULL;
		         lc->suiv      = l;
		       } 
		   }
              }
         }
    }
};



/* Insere un element ligne de la matrice. */

void Inserer_Ligne(int som,double val,int succ)

{
     struct liste_sommets   *lc, *l; 
     int                     finie;

    l  = NULL;
    lc = mat[som].lig;
    if(mat[som].lig == NULL)
      {
	if((mat[som].lig= (struct liste_sommets *) 
                          calloc (1, sizeof(struct liste_sommets))) == NULL)
          exit(1);
	else
	  {
	    mat[som].lig->som_global = succ;
            mat[som].lig->val        = val;
            mat[som].lig->suiv       = NULL;
	  }
      }
    else /* il y a au moins un element dans la ligne */ 
      {
       if (lc->som_global == succ)
          lc->val = lc->val + val;
       else /* lc->som_global != succ */
         {
           if(lc->som_global >  succ)
             {
	       if((l= ( struct liste_sommets *) 
                             calloc (1 , sizeof(struct liste_sommets)))==NULL) 
                { 
                  exit(1);
                }
	       else
	         {
                   l->som_global = succ;
                   l->val        = val;
		   l->suiv       = lc;	      
                   mat[som].lig  = l;
	         }
              }
            else 
              {
	        finie =0;		
                while ((lc->suiv != NULL) && (finie==0)) 
	          {
		    if(lc->suiv->som_global  < succ)
		      lc = lc->suiv;
		    else 
                      finie = 1;
		  } 
                if(lc->suiv != NULL) 
		  {
		    if (lc->suiv->som_global == succ)
                       lc->suiv->val = lc->suiv->val +  val;
		    else 
                      {
                        if ((l = ( struct liste_sommets *) 
                             calloc (1, sizeof(struct liste_sommets))) == NULL)
			   exit(1);    
			else
			  {        
             		    l->som_global = succ;
              		    l->val        = val;
              		    l->suiv       = lc->suiv;
			    lc->suiv       = l;
			  } 
                      }
		   }
                 else 
                   {                       
		     if((l = ( struct liste_sommets *) 
                         calloc (1, sizeof(struct liste_sommets))) == NULL)
		        exit(1);
		     else
		       {
                         l->som_global = succ;
                         l->val        = val;
		         l->suiv       = NULL;
		         lc->suiv      = l;
		       } 
		   }
              }
         }
    }
};

/* -- lire une chaine dans un buffer  --*/
 void lire_chaine(char **deb,char nombre[25]) 
 {
 char *fin;
 fin=*deb;
 if (*fin==' ')
  	for ( ; *fin==' ';fin++); 

 for ( ; (*fin!=' ')&(*fin!='\0');fin++); 
 /* on saute le nombre et fin pointe sur la fin du nombre */
 
 sscanf(*deb,"%s",nombre);
 
 for ( ; *fin==' ';fin++); 
 /* on saute les blancs et fin pointe sur le debut du nombre */
 *deb=fin; 
 return;
 }	
/* -- ecrire un double dans une chaine a la suite --*/
 void ecrire_chaine(char **deb,double nombre) 
 {
 char fin[25];
 sprintf(fin," %.14e\n",nombre);
 strcat(*deb,fin);
 return;
 }	


/* -- Recupere la matrice du fichier et en fait une copie en memoire */

void  Construire_Matrice_Transition(char *fich) 
{
	int    som, nzero, col, nbnozero,i;   
	double val;
	int nmax=50000;
	int ns=20;
	char nombre[25];
	char *deb; /* declaration pour lire_chaine */
	char *ligne_matrice; /* buffer d'une ligne de la matrice */
    char s0[200];
    char s1[200];
    char s2[200];

    
    strcpy(s0,fich);
    strcpy(s1,s0);
    strcat(s1,".sz");
    strcpy(s2,s0);
    strcat(s2,".Rii");

    	/* printf("lecture de la taille du systeme, puis de la matrice de transition rangee par lignes \n");*/
	if ((G=fopen(s1,"r"))==NULL)
        {
          puts("Le fichier   .sz n'existe pas \n") ;
            puts(s1);
          exit(1) ;
	}  
	
	/* fscanf(G,"%d",&nbnozero);
	printf("nbnozero = %d\n",nbnozero); */
	fgets(nombre,ns,G);

	nbnozero=atoi(nombre);
	printf("nbnozero = %d\n",nbnozero);
	/* fscanf(G,"%d",&N); */
	fgets(nombre,ns,G);
	N=atoi(nombre);
	printf("N = %d\n",N);
	
    fclose(G);
    
    if ((G=fopen(s2,"r"))==NULL)
    {
        puts("Le fichier .Rii  n'existe pas \n") ;
        exit(1) ;
	}

        /* Recuperation de la matrice creuse binaire */

	if((mat= ( struct matgth *) calloc ( (N+1) , sizeof(struct matgth)))==NULL) {printf("pas de mem \n");exit(1);}
	if((pi= (double *) calloc ((N+1) , sizeof(double)))==NULL) {printf("pas de mem \n");exit(1);}
	
	/* pour lire une ligne complete du fichier matrice */
	if((ligne_matrice=(char *)calloc((N+1)*25,sizeof(char)))==NULL) {printf("pas de mem \n");exit(1);}
    
 	for (som= 1;som<=N;som++)
        {  
     		S    = 0.0;
		fgets(ligne_matrice,nmax,G);
	/*	printf("longueur de la ligne %d\n",strlen(ligne_matrice)); */
		deb=ligne_matrice;
		lire_chaine(&deb,nombre) ;  		
    /* on jette le numro de ligne */ 
        lire_chaine(&deb,nombre) ;
 		nzero=atoi(nombre);
     		for (i=1;i<=nzero;i++) 
         	{
			lire_chaine(&deb,nombre);  
			val=atof(nombre);
			lire_chaine(&deb,nombre); 
			col=atoi(nombre);
		/*	printf ("%.14e %d\n",val,col); */
            		if  (col+1 < som)
                            {		    
             		    Inserer_Ligne(som,val,col+1);
                  	    S  = S + val;
                            }
             		else Inserer_Colonne(col+1,val,som);
     	 	}    
     		mat[som].somme = S;   
       	}
	fclose(G);
	if(!ligne_matrice)
	free(ligne_matrice);
}


/*-- Des Calculs pour Algorithme GTH  */

void Calculer(struct liste_sommets  *ll2,int som,double val)

{
       struct liste_sommets   *l , *l1, *l2, *preced;
       int                    fini;
       
l      = NULL;       
l1     = mat[som].lig;
l2     = ll2;
preced = NULL;

while ((l2 != NULL) && (l1 != NULL))
   {
         if ((l2->som_global) < som)
          {
              if ((l1->som_global) > (l2->som_global))
               { 
                 if ((l=(struct liste_sommets *) 
                       calloc (1, sizeof(struct liste_sommets))) == NULL)
		    { exit(1);}
	         if (preced != NULL)
                    {
                     l->suiv       = l1;
                     preced->suiv  = l;
                     preced        = l;
                    }
                 else 
                    {
                     l->suiv       = l1;
                     preced        = l;
                     mat[som].lig  = l;
                    }
                 l->som_global   = l2->som_global;
                 l->val          = l2->val*val;
                 mat[som].somme = mat[som].somme + l->val;  
                 l2 = l2->suiv;
	       } 
              else
	        {
                 if (l1->som_global ==  l2->som_global)  
                    {
                     l1->val += l2->val *val;
                     mat[som].somme+= l2->val *val; 
                     preced = l1;
	             l1 = l1->suiv; 
	             l2 = l2->suiv;
                    }
                 else 
		    { 
		     preced = l1;
	             l1 = l1->suiv; 
		    }
                }
          }
	else
	   {
  	    if (l2->som_global == som) 
               l2 = l2->suiv;
            else
	      {
		fini = 0;
                while ((l2 != NULL) && (fini ==0))
	          { 
	           if (l2->som_global > som)
		      { 
                      Inserer_Colonne(l2->som_global, (l2->val)*val, som);
                      l2 = l2->suiv;
		      }
		   else fini = 1;   
                  } /* Fin du while */
	      }

	   }
                 
   } /* Fin du while */

if ((l2 != NULL) && (l1 == NULL))
      {
	fini = 0;      
        while ((l2 != NULL) && (fini == 0))
          {    
	     if (l2->som_global < som)
	      {	                    
               if ((l = (struct liste_sommets *) 
                         calloc (1, sizeof(struct liste_sommets))) == NULL)
	          { exit(1);}
	       l->val        = (l2->val)*val;
               l->som_global = l2->som_global;
               mat[som].somme = mat[som].somme + l->val;                     
               if (preced != NULL) 
                  preced->suiv = l;
               else 
                 {
                   mat[som].lig = l; 
                   l1 = l;
                 }
               preced = l;
               l->suiv = NULL;                     
               l2 = l2->suiv; 	          
	      }
	     else fini = 1;                     
           } /* Fin du while */

               if (l2 != NULL) 
	          { if (l2->som_global == som) l2 = l2->suiv; }
               fini=0;
               while ((l2 != NULL) && (fini ==0))
                    {
	             if (l2->som_global > som)
                      {
	              Inserer_Colonne(l2->som_global, (l2->val)*val, som);
                      l2 = l2->suiv;
		      }
		     else fini = 1; 
                    }  /* Fin du while */            
      }
}
             

/* -- DEBUT DU MAIN */


int main(int argc,char **argv)
/*  DEBUT DU MAIN */


{ 
	    
  int fine;

if(argc !=3 )
 {
  puts("bad args ");
  exit(1);
 }

      printf("\n\n\n ******************************************************************\n");
      printf(" ****         Calcul de la distribution stationnaire           ****\n");
      printf(" ****                  par l'algorithme GTH                    ****\n");
      printf(" ****     La matrice en entree est une matrice creuse texte    ****\n");
      printf(" ******************************************************************\n\n");
	
      Construire_Matrice_Transition(argv[2]);
      avant1_user = time(&avant_user);
      printf("\n\n           RUN...\n\n\n"); 
      for (j=N; j>=2; j--)
       {    
            S = 0.0;  
            llc = mat[j].col;
	    fine = 0;
            while ((llc != NULL)  && (fine ==0))
  	       {       
		  if  (llc->som_global < j)
		   {
	            if (mat[j].somme == 0)
		     { printf(" PROBLEME dans la chaine.\n\n\n");
		       printf("           ARRET DU TRAITEMENT...\n\n\n");
		       exit(1);
		     }  	     	     		                               
                    llc->val = (llc->val / mat[j].somme);
                    ll = mat[j].lig;
                    Calculer(ll, llc->som_global, llc->val);              
                    llc     = llc->suiv;
		   }
		  else fine = 1;   
               }
      } 
/* --propagation de la solution X1 pour trouver X2.*/

      pi[1] = 1.0;
      norm  = 1.0;
      for (j=2; j<=N; j++)
        {
          llc = mat[j].col;
          if (llc != NULL)  
	   { if (llc->som_global == 1) { pi[j] = llc->val; llc = llc->suiv;} } 
          fine =0;
	  while ((llc != NULL) &&(fine==0))	      
           {  
		if (llc->som_global < j) 
		 {  
                  pi[j] = pi[j]+ pi[llc->som_global]*llc->val;
                  llc = llc->suiv;
		 }
	        else fine =1;	  
           }
          norm = norm + pi[j];
        }
	printf(" Le phase de calcul dure :");
       Affich_clock();
/* -- Ecriture de la distribution stationnaire dans un fichier */
	if((fic_resultat= (char *) calloc ( (20) , sizeof(char)))==NULL) {printf("pas de mem \n");exit(1);}
	strcpy(fic_resultat,argv[2]);
	strcat(fic_resultat,".pi");
	G=fopen(fic_resultat,"w");
       /* G = fopen("Pi_Gth","w");*/
        printf("   %d states ", N);
        S = 0.0;
	
	if ((buffer_pi= (char *) calloc ( (300) , sizeof(char)))==NULL) {printf("pas de mem \n");exit(1);}
        pt_buffer_pi = buffer_pi ;
    	for (j=1;j<=N;j++)
         {
   	    pi[j]=pi[j]/norm; 
	    ecrire_chaine(&pt_buffer_pi,pi[j]);
	    
	    if (j % 10 == 0) /* on ecrit tous les 10 probas */
	    	{
            	fprintf(G,"%s",buffer_pi);
        if(!buffer_pi)
		free(buffer_pi);
		if ((buffer_pi= (char *) calloc ( (300) , sizeof(char)))==NULL) {printf("pas de mem \n");exit(1);}
        	pt_buffer_pi = buffer_pi ;
		}
            S = S + pi[j];
         }
	if (N%10 != 0) 
		fprintf(G,"%s",buffer_pi); /*on vide le buffer */
		if(!buffer_pi)
	    free(buffer_pi);
        printf(" and the sum of probas is  %E\n",S);
        fclose(G);

       printf(" Le temps total est :");
       Affich_clock();
 
       printf("\n\n La Distribution Stationnaire est dans le fichier %s \n\n",fic_resultat); 
       printf("\n           FIN DE TRAITEMENT\n\n");
       
};

/* FIN DU MAIN */


