/* 
GTH en matrice pleine au format Xborne en entree
needs .Rxx matrix and .sz size file
version long double modifiee pour lire toutes 
les matrices Rxx
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void  usage()
{
	printf("usage : Gth -f filename Suffix \n");
	printf("filename.Suffix and filename.sz must exist before. And the suffix must be Rxx \n");
	exit(1);
}

void  ProblemeMemoire()
{
	printf("Sorry, not enougth memory for the matrix and the probability vectors \n");
	exit(1);
}


int main(argc, argv)
int argc;
char *argv[];
{
  int NSommets, NArcs;
  int i, j, n, k, c, jx, degre;
long  double Tot, S, x;
  FILE *pf1;
  char s0[20];
  char s1[20];
  char s2[20];
  char s3[20];

/* les objets dynamiques a creer par malloc */
 
long    double *P;
long    double *pi;

/*
On recupere le nom du modele et on verifie que les fichiers existent 
*/

	if (argc!=4) usage();
	--argc;
	++argv;
	if (**argv != '-') usage();
	switch (*++*argv) {
		case 'f' :  /* on recupere le nom de fichier */
		{++argv;
		strcpy(s0,*argv);
		++argv; /* on recupere le nouveau suffixe */
		strcpy(s1,*argv);
		/* on ajoute .sz, et le suffixe */
		strcpy(s2,s0);
		strcat(s2,".sz");
		strcpy(s3,s0);
		strcat(s3,".pi");
		/* on verifie si il existe un fichier .sz*/
		if ((pf1=fopen(s2,"r"))==NULL)
			{
			usage();
			}
		/* on verifie si le suffixe est Rxx */
		if (s1[0]!='R')  
		    	{
			usage();
			}
		/* on verifie que le fichier model.suffix existe */
		strcat(s0,".");
		strcat(s0,s1);
		if ((pf1=fopen(s0,"r"))==NULL)
			{
			usage();
			}
		break;
		}
		default	 : usage();
	}

/*
On recupere les tailles dans le fichier filename.sz
*/
  pf1=fopen(s2,"r");
  fscanf(pf1,"%d\n", &NArcs);
  fscanf(pf1,"%d\n", &NSommets);
  fclose(pf1);	
  /*printf("%d\n", NArcs);
  printf("%d\n", NSommets);*/

/* 
On cree les objets ou on exit(1) en cas de pb
*/

  if (!(pi=(long double *)malloc(NSommets*sizeof(long double)))) ProblemeMemoire();
  if (!(P=(long double *)calloc(NSommets*NSommets, sizeof(long double)))) ProblemeMemoire();
  
/*
Lecture de la matrice dans le fichier filename.dt
*/
  pf1=fopen(s0,"r");
  for (i = 0; i < NSommets; i++) {
  /* 
  on commence par le numero et le degre
  */
    fscanf(pf1,"%d", &c);
    fscanf(pf1,"%d", &degre);
    for (j = 0; j < degre; j++) {
      fscanf(pf1,"%Lf", &x);
      fscanf(pf1,"%d", &jx);
/*  printf("Lect %ld %ld %ld %lf \n", c,degre, jx, x);  */
      P[c*NSommets+jx]=x; /* P[c,jx]=x */
    }
    getc(pf1);
  }
  fclose(pf1);

  for (n = NSommets-1; n >= 1; n--) {
    S = 0.0L;
    for (j = 0; j < n; j++) {
      S +=P[n*NSommets+j];  /* P[n,j]; */
/*  printf("Pr %lf %lf\n", P[n*NSommets+j], S); */
      }
    if (S == 0.0) {
      printf("Probleme en %d %Lf\n", n,S);
    exit(1);
    }
      for (i = 0; i < n; i++)
      P[i*NSommets+n] /= S; /* P[i,j]; */
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) 
      	P[i*NSommets+j] += P[i*NSommets+n] * P[n*NSommets+j];
    }
  }
  Tot = 1.0L;
  pi[0] = 1.0L;
  for (j = 1; j < NSommets; j++) {
    pi[j] = P[j];  /*  P[0,j] */
    for (k = 1; k < j; k++)
      pi[j] += pi[k] * P[k*NSommets+j]; /* P[k][j] */
    Tot += pi[j];
  }
  for (j = 0; j < NSommets; j++) pi[j] /= Tot;
    
  pf1=fopen(s3,"w");
  for (i = 0; i < NSommets; i++) fprintf(pf1," %.18Le\n", pi[i]);
  fclose(pf1);
  //printf("Done GTHLD Row\n");
  exit(0);
}
