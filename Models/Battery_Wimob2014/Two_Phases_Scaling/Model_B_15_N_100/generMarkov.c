/*
 generation des etats a partir d'une equation d'evolution
 V1.5 en C, le 1/3/14
 
 Hypotheses de fonctionnement 
 
 - La taille maximale est le produit des tailles des composantes 
 - elle est calculee en fonction du modele 
 - deux structures de tableau sont dimensionnees dynamiquement a cette taille
 - si impossible d'allouer le tableau, exit 1.
 - on ne genere que des etats atteignables   
 - on utilise une fonction de numerotation construite a partir 
 des tailles des composantes. Cette fonction doit rendre un 
 entier. Elle utilise les deux tableaux de taille maximale. 
 Il y a risque de debordement meme si il y peu d'etats
 atteignables. 
 - matrice creuse : on insere les elements un par un dans une 
 liste triee: ceci n'est efficace que si il y a peu d'etats par 
 ligne. Sinon il faudrait ameliorer en changeant la structure 
 de donnees
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>  // Pour getpid()

/* 
 inclusions des constantes, types et variables particulieres a chaque modele
 */
#include "const.h"
#include "var.h"

#define DEBUG 0

/*Variables donnant l'etendue de chaque variable isolee*/

static int Min[NEt], Max[NEt], Deadline, Buffer;
static double seuil;

/* Pour  les etats  du modele */ 

typedef int Etat[NEt];

/*On utilise cet objet pour garder les transitions probabilises de la
 chaine et aussi pour garder les etats en cours de creation*/

struct item {
	Etat voisin;
	double prob;
	int numero;
	struct item *next, *last;
};

typedef struct item item;

/*Variables qui doivent etre invisibles pour le pgm principal*/
/* NE PAS TOUCHER */

item *Poubelle;   /*pointeur vers le sommet de la pile poubelle*/
item *APrendre, *PourAjouter; /*pointeurs sommets en attente de visite*/
item *PointeVisite;   /*pointeur  vers le sommet des successeurs*/

int succ; /* le degre du sommet courant  */
int NbSommets; /* le nombre de sommets atteignables */
int NbArcs; /* le nombre de transitions */

/* Les deux structures  de grande taille dimensionnees par Maxsize */
int MaxSize;
int NombreDeMark; 
int *Mark; //short
int  *TableCorrespondance;

/*---------------------------------------------*/
/* conversion d'un etat en un numero           */
/*  attention, il est possible que cela deborde*/
/*  le numero minimal est 1                    */
/*---------------------------------------------*/

int Etat2Int(E)
int *E;
{
	int i, j;
	i = E[0] - Min[0];
	for (j = 1; j < NEt; j++)
    {
		i = i * (Max[j] - Min[j] + 1) + E[j] - Min[j];
    }
	return (i + 1);
}

/*---------------------------------------------*/
/*gestion usuelle d'une pile de vieux elements */
/*pour eviter d'employer la function de        */
/*liberation memoire                           */
/*---------------------------------------------*/
void InitPoubelle()
{
	Poubelle = NULL;
	PourAjouter = NULL;
	APrendre = NULL;
}

static void Nouveau(p)
item **p;
{
	if (Poubelle != NULL) {
		*p = Poubelle;
		Poubelle = ((*p)->next);
	} else
		*p = (item *)malloc(sizeof(item));
}

void Depose(p)
item *p;
{
	p->next = Poubelle;
	Poubelle = p;
}

/*---------------------------------------------*/
/*                                             */
/*Gestion de la liste des successeurs d'un etat*/
/*avec ses probas de transition                */
/*Ajoute Retire InitListeSuccesseurs           */
/*                                             */
/*---------------------------------------------*/

int inferieurEtat(X, Y)
int *X, *Y;
{
	int i, j, k, l;
	i = Etat2Int(X)-1;
	j = Etat2Int(Y)-1;
	k = TableCorrespondance[i];
	l = TableCorrespondance[j];
	if  (k<l) {return(0);}
	return(1);
}

static int egalEtat(X, Y)
int *X, *Y;
{
	int i, j;
	i = Etat2Int(X);
	j = Etat2Int(Y);
	if (i == j) {return(0);}
	return(1) ;
}

// Version qst
item  *InsereEtat(debut,p,G)
item *debut;
double p;
Etat   G;
{
	item *a_inserer;
	int i;
	
	if (debut == NULL) {
		Nouveau(&debut); 
		debut->prob = p;
		debut->next = NULL;
		i = Etat2Int(G) -1;
		debut->numero=TableCorrespondance[i];
		succ++;
		for (i=0; i<NEt; i++) {debut->voisin[i] = G[i];} 
		return debut;
	}
	if (egalEtat(debut->voisin,G)==0) {
		debut->prob += p;
		return debut;
	}
	if (inferieurEtat(debut->voisin, G)==0) {
		debut->next = InsereEtat(debut->next,p,G);
		return debut;
	}
	
	Nouveau(&a_inserer);
	for (i=0; i<NEt; i++) {a_inserer->voisin[i]=G[i];}
	a_inserer->prob = p;
	a_inserer->next = debut;
	i = Etat2Int(G)-1;
	a_inserer->numero=TableCorrespondance[i];
	succ++;
	return a_inserer;
}

/*gestion de la liste des successeurs du sommet courant */
/*elle est effectivement implementee sous forme de liste triee*/

void Ajoute(p, F)
double p;
Etat F;
{
	PointeVisite = InsereEtat(PointeVisite,p,F);
}

void Retire(It)
item *It;
{
	/*On rend le premier element de la liste*/
	item *q;
	
	if (PointeVisite == NULL)
    {
		fprintf(stderr,"PointeVisite = NULL\n");
		exit(7);
    }
	q = PointeVisite;
	PointeVisite = PointeVisite->next;
	*It = *q;
	Depose(q);
}


int NbVoisins()
{
	return succ;
}


void InitListeSuccesseurs()
{
	/*initialisation des variables liees a la liste des succeseurs du sommet courant*/
	succ = 0;   /*nombre de successeurs distincts du sommet courant*/
	PointeVisite = NULL;
}


/* Inclusion des fonctions specifiques a un modele */

#include "fun.c"

/*---------------------------------------------*/
/*                                             */
/*Gestion des sommets deja visites et a visiter*/
/*  MiseEnFile  FileVide SortieDeFile          */
/* NumeroDeVisite, CreerMark, InitMark         */
/*                                             */
/*---------------------------------------------*/
void MiseEnFile(It)
item It;
{
	item *p;
	
	Nouveau(&p);
	*p = It;
	
	p->next = PourAjouter;
	p->last = NULL;
	if (PourAjouter == NULL)
		APrendre = p;
	else
		PourAjouter->last = p;
	PourAjouter = p;
}

int FileVide()
{
	if (PourAjouter == NULL) {return 1;} 
	return(0);
}

void SortieDeFile(It)
item *It;
{
	item *p;
	
	*It = *APrendre;
	p = APrendre;
	APrendre = APrendre->last;
	if (APrendre == NULL)
		PourAjouter = NULL;
	Depose(p);
}

void  ProblemeMemoire()
{
	printf("Sorry, not enougth memory for the product of the components, \n");
	printf("try to change your model decomposition or size of the components \n");
	exit(1);
}


void CreerMark()
{
	int i, j;
	i = 1;
	for (j = 0; j < NEt; j++)
		i *= (Max[j] - Min[j] + 1);
	i++;
	MaxSize=i;
	/*
	 i est la taille des deux structures que l'on doit creer 
	 */
	if (!(Mark=(int *)calloc(MaxSize, sizeof(int)))) ProblemeMemoire();
	printf("First Malloc OK, vector of size %12d\n", MaxSize);
	if (!(TableCorrespondance=(int *)calloc(MaxSize,sizeof(int)))) ProblemeMemoire();
	printf("Second Malloc OK, same size \n");
	
}


void InitMark()
{	
	NombreDeMark = -1;
}


void PrintEtat(E)
Etat E;
{
	int i;
	for(i=0;i<NEt;i++) {
		printf("%5d",E[i]);
	}
    printf("\n");
}

int DejaVisite(It)
item It;
{
	int i;
	int deja;
	i = Etat2Int(It.voisin)-1;
	deja = Mark[i];
	if (Mark[i]==1) Mark[i]=2;
	return deja;
}

int NumeroDeVisite(E)
Etat E;
{
	int i,j;
	i = Etat2Int(E)-1;
	/*  printf("%12d \n",i);   PrintEtat(E); */
	if (Mark[i]>0)
    {
		j = TableCorrespondance[i];
    }
	else {
		Mark[i]=1;
		NombreDeMark++;
		TableCorrespondance[i] = NombreDeMark;
		j = NombreDeMark;
	}
	NbSommets = NombreDeMark + 1;
	return(j);
}



/* Programme principal et interface exterieure */


void  usage()
{
	printf("usage : GenerMatrix -f filename //seuil deadline BufferSize \n");
	printf("to create filename.Rii, filename.cd and filename.sz \n");
	printf("to store the description of the states and the matrix of your model \n");
	printf("The files must not exist before. \n");
	exit(1);
}


int main(argc,argv)
int argc;
char *argv[];
{
	Etat F, G, R;
	item It, It2;
	int j, k, bcli;
	double SommeDeProba;
	int deja;
	double pro, pro1, pro2;
	FILE *pf1, *pf2;
	char s1[20];
	char s2[20];
	char s3[20];
	/*double seuil;
	int Deadline;*/
	
	if (argc<6) usage();
	--argc;
	++argv;
	if (**argv != '-') usage();
	switch (*++*argv) {
		case 'f' :  /* on recupere le nom de fichier */
		{++argv;
			strcpy(s1,*argv);
			/* on ajoute .sz, .cd ou .Rii */
			strcpy(s2,s1);
			strcpy(s3,s1);
			strcat(s1,".cd");
			strcat(s2,".Rii");
			strcat(s3,".sz");
			/* on verifie si il existe deja */
			if ((pf1=fopen(s1,"r"))!=NULL)
			{
				fclose(pf1);
				usage();
			}
			if ((pf1=fopen(s2,"r"))!=NULL)
			{
				fclose(pf1);
				usage();
			}
			if ((pf1=fopen(s3,"r"))!=NULL)
			{
				fclose(pf1);
				usage();
			}
			
			seuil = atof(*(++argv));
			Deadline =atoi (*(++argv));
			Buffer = atoi (*(++argv));
			//seuil /= Buffer;
			printf("seuil = %.2f%% et deadline = %d \n",(seuil/Buffer)*100,Deadline);
			pf1=fopen(s1,"w");
			pf2=fopen(s2,"w"); 
			break;
		}
		default	 : usage();
	}

	srand(time(NULL) * getpid());
	InitPoubelle();
	InitEtendue(Deadline,Buffer);
	InitParticuliere();
	CreerMark();
	InitMark();
	NbArcs = 0;
	NbSommets = 0;
	EtatInitial(It.voisin);
	It.next = NULL;
	It.last = NULL;
	It.prob = 0.0;
	It.numero = NumeroDeVisite(It.voisin);
	j = Etat2Int(It.voisin)-1;
	Mark[j]=2;
	MiseEnFile(It);
	
	do {
		SortieDeFile(&It);
		InitListeSuccesseurs();
		
		/*on cherche tous les successeurs distincts de It.voisin en generant 
		 tous les evenements et en les appliquant a cet etat. 
		 L'etat resultat est F. On ajoute F dans la liste des 
		 successeurs de l'etat en cours*/
		
#if (Polynom==0) 
		/* One step transition  P */
		SommeDeProba = 0.0;
		for (j = 1; j <= NbEvtsPossibles; j++) {
#if (DEBUG==1)			
			printf("j=%d\n",j);
			PrintEtat(It.voisin);
#endif
			Equation(It.voisin, j, F, R,seuil,Deadline);
#if (DEBUG==1)
			PrintEtat(F);
#endif
			pro = Probabilite(j, It.voisin);
			SommeDeProba+=pro;
#if (DEBUG==1)
			printf("Proba : %lf\n", pro);
			printf("SommeDeProba : %lf\n", SommeDeProba);
#endif
			if (pro>0) {bcli=NumeroDeVisite(F); Ajoute(pro, F);}
		}
#endif 
		
		
#if (Polynom==1) /* Id/2 + P/2 */
		SommeDeProba = 0.5;
		bcli=NumeroDeVisite(It.voisin); 
		Ajoute(0.5, It.voisin);
		for (j = 1; j <= NbEvtsPossibles; j++) {
#if (DEBUG==1)			
			printf("j=%d\n",j);
			PrintEtat(It.voisin);
#endif
			Equation(It.voisin, j, F, R,seuil,Deadline);
#if (DEBUG==1)
			PrintEtat(F);
#endif
			pro = Probabilite(j, It.voisin)/2;
			SommeDeProba+=pro;
#if (DEBUG==1)
				printf("Proba : %lf\n", pro);
        printf("SommeDeProba : %lf\n", SommeDeProba);
#endif
			if (pro>0) {bcli=NumeroDeVisite(F); Ajoute(pro, F);}
    }
#endif 
	
#if (Polynom==2)  /* Id/2 + P^2/2 */
		SommeDeProba = 0.5;
		bcli=NumeroDeVisite(It.voisin); 
		Ajoute(0.5, It.voisin);
		for (j = 1; j <= NbEvtsPossibles; j++) {
#if (DEBUG==2)			
			printf("j=%d\n",j);
			PrintEtat(It.voisin);
#endif
			Equation(It.voisin, j, F, R,seuil,Deadline);
#if (DEBUG==2)
			PrintEtat(F);
#endif
			pro1 = Probabilite(j, It.voisin);
			for (k = 1; k <= NbEvtsPossibles; k++) {
				Equation(F, k, G, R,seuil,Deadline);
#if (DEBUG==2)
				PrintEtat(G);
#endif
				pro2 = Probabilite(k, F);
				pro = pro1 * pro2 /2.0; 
				SommeDeProba+=pro;
#if (DEBUG==1)
				printf("Proba : %lf\n", pro);
				printf("SommeDeProba : %lf\n", SommeDeProba);
#endif
				if (pro>0) {bcli=NumeroDeVisite(G); Ajoute(pro, G);}
			}
		}
#endif 

		
#if (Polynom==3)  /* P^2 */
		SommeDeProba = 0.0;
		for (j = 1; j <= NbEvtsPossibles; j++) {
#if (DEBUG==2)			
			printf("j=%d\n",j);
			PrintEtat(It.voisin);
#endif
			Equation(It.voisin, j, F, R,seuil,Deadline);
#if (DEBUG==2)
			PrintEtat(F);
#endif
			pro1 = Probabilite(j, It.voisin);
			for (k = 1; k <= NbEvtsPossibles; k++) {
				Equation(F, k, G, R,seuil,Deadline);
#if (DEBUG==2)
				PrintEtat(G);
#endif
				pro2 = Probabilite(k, F);
				pro = pro1 * pro2; 
				SommeDeProba+=pro;
#if (DEBUG==1)
				printf("Proba : %lf\n", pro);
				printf("SommeDeProba : %lf\n", SommeDeProba);
#endif
				if (pro>0) {bcli=NumeroDeVisite(G); Ajoute(pro, G);}
			}
		}
#endif 

		
	
    /*ecriture du codage des etats sur les composantes*/
	
    NbArcs += NbVoisins();
    fprintf(pf1,"%12d", It.numero);
    for (j = 0; j < NEt; j++)
		fprintf(pf1,"%12d", It.voisin[j]);
    fprintf(pf1,"\n");
	
    /*ecriture du numero de sommet puis de son degre */
	
    fprintf(pf2,"%12d", It.numero);
    bcli = NbVoisins();
    fprintf(pf2,"%12d", bcli);
    SommeDeProba = 0.0;
    for (j = 1; j <= bcli; j++) {
		Retire(&It2);
		/*On examine la liste des successeurs pour poursuivre les visites */
		deja=DejaVisite(It2);
		if (deja<2)  MiseEnFile(It2);
		/* 
		 pour chaque successeur, on ecrit la probabilite de transition 
		 puis le numero de l'etat
		 */
		fprintf(pf2,"% .15E%12d", It2.prob, It2.numero);
		SommeDeProba += It2.prob;
    }
    fprintf(pf2,"\n");
    /*
	 on passe a la ligne pour la ligne suivante de la matrice
	 */
    if (fabs(SommeDeProba - 1.0) > 1.0e-9)
	{
		printf("attention la somme vaut : % .10E\n", SommeDeProba);
		exit(0);
	}

} while (!FileVide());




fclose(pf2);
fclose(pf1);
pf1=fopen(s3,"w"); 
fprintf(pf1,"%12d\n", NbArcs);
fprintf(pf1,"%12d\n", NbSommets);
fprintf(pf1,"%12d\n", NEt);
fclose(pf1);	
exit(0);
}

