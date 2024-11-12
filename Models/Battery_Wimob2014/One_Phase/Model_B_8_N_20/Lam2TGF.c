/* 
 Transformation des fichiers d'un modele en donnees TGF pour dessin du graphe 
 
 V1.0 le 20/12/12
 
 Hypotheses  de fonctionnement : 
 l'argument est le nom du modele : filename 
 la taille est dans le fichiez filename.sz
 le codage dans filename.cd, 
 et la matrice dans filename.Rii
 */

#include <stdio.h> 
#include <math.h>
#include <string.h>
#include <stdlib.h>

void usage() { 
    printf("usage : Lam2TGF -f filename \n");
    printf("filename.Rii, filename.sz and filename.cd must exist before \n");
    exit(1); 
}

void ProblemeMemoire() { printf("Sorry, not enougth memory \n"); exit(1); }


int main(argc, argv) 
int argc; 
char *argv[]; 
{ 
	/* Le nombre d'etats atteignables et de transitions */ 
	long NArcs;
	int ReachableStates;
	
	FILE *pf1, *pf2; 

    int i,j, ji, k, nb, degre, NEt, dest ; 
	float prob;
    char s1[20]; 
    char s2[20]; 
    char s3[20]; 
    char s4[20];
	
	/* On recupere le nom du modele et on verifie que les fichiers existent */
	
    if (argc!=3) usage();
    --argc; ++argv; 
    if (**argv != '-') usage(); 
    switch (*++*argv) { 
		case 'f' : { 
			/* on recupere le nom de fichier */
			++argv;
			strcpy(s1,*argv); 
			/* on ajoute .sz, .Rii et .cd */ 
			strcpy(s2,s1);
			strcpy(s3,s1); 
			strcpy(s4,s1); 
			strcat(s1,".sz"); 
			strcat(s2,".Rii");
			strcat(s3,".cd"); 
			strcat(s4,".tgf"); 
			/* on verifie si ils existent */ 
			if ((pf1=fopen(s1,"r"))==NULL) usage(); 
			if ((pf1=fopen(s2,"r"))==NULL) usage(); 
			if ((pf1=fopen(s3,"r"))==NULL) usage(); 
			break; 
		} 
		default : usage(); 
    } 
	
	/* On recupere les tailles dans le fichier filename.sz */
    pf1=fopen(s1,"r"); 
    fscanf(pf1,"%12ld\n", &NArcs); 
    fscanf(pf1,"%12d\n",&ReachableStates); 
	fscanf(pf1,"%d\n",&NEt); 
	
    printf("%12ld\n", NArcs); 
    printf("%12d\n",ReachableStates); 
    printf("%12d\n",NEt); 	
    fclose(pf1); 
	
	
    printf("debut lecture code \n"); 
    /* Lecture de la matrice de codage dans le fichier filename.cd */
    pf1=fopen(s3,"r"); 
	pf2=fopen(s4,"w"); 
    for (k = 0; k < ReachableStates; k++) {
		fscanf(pf1,"%d", &j);
		fprintf(pf2,"%d  (", j);
		for (i = 0; i < NEt-1; i++) {
			fscanf(pf1,"%d",&nb);
			fprintf(pf2,"%d,", nb);
		} 
		fscanf(pf1,"%d",&nb);
		fprintf(pf2,"%d)\n", nb);
		getc(pf1);
   }
    fclose(pf1);
	fprintf(pf2,"# \n ");		
    printf("fin lecture code \n"); 
	
	
    /*   Lecture de la matrice dans le fichier filename.Rii */
    pf1=fopen(s2,"r");
	for (i = 0; i < ReachableStates; i++) {
		fscanf(pf1,"%d", &ji);
		fscanf(pf1,"%d", &degre);
		for (j = 1; j <= degre; j++)  {
			fscanf(pf1,"%g", &prob);
			fscanf(pf1,"%d", &dest);
			fprintf(pf2, "%d %d \n", ji, dest);
			getc(pf1);    
		}
	}
	fclose(pf1);		
	printf("fin lecture Rii \n"); 
	fclose(pf2);
	printf("Done Lam2TGF\n");
	exit(0); 
}


