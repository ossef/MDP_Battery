#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <unistd.h>

//#define DEBUG 1
// #define DEBUG2

#define VERSION 1  // V1 : pour matrice carree et V2 : pour matrice non carree

// V2 : accepts non square matrix Made by qst dec 2015
// V1 : Made by qst sept 2003

char *NOM_FIC_IN;
char *NOM_FIC_OUT;
char *EXT_IN;
char *EXT_OUT;
char *NOM_BASE;
char *NOM_FIC_sz;
char *NOM_ECLATE;
char *NOM_TRI;

unsigned long long int nb_lignes, nb_colonnes, nb_vals, nb_automates;
unsigned long long int min_l, max_l, min_c, max_c;

unsigned long long int *deg_l, *deg_c;

void usage(int argc, char* argv[]) {
	fprintf(stderr,"usage: %s file.extin extout\n",argv[0]);
	fprintf(stderr,"\t extin and extout must be {R|C}{i|d|u}{i|d|u}\n");
	fprintf(stderr,"\t output is written to file.extout\n");
	fprintf(stderr,"\t example: %s modele.Rii Cdi produces modele.Cdi\n",argv[0]);
	exit(1);
}

// Vérifie que l'extension en argument respecte les formats acceptables
int extension_valide(char *ext) {
	if (strlen(ext) != 3) return 0;
	if ( (ext[0] != 'R') && (ext[0] != 'C') ) return 0;
	if ( (ext[1] != 'i') && (ext[1] != 'd') && (ext[1] != 'u') ) return 0;
	if ( (ext[2] != 'i') && (ext[2] != 'd') && (ext[2] != 'u') ) return 0;
	return 1;
}

// Calcule de tous les noms d'entrées et de sortie
// et vérifie que ces noms sont corrects
void calcul_noms(int argc, char* argv[]) {
	// Pas le bon nombre d'arguments
	if (argc!=3) usage(argc,argv);
	int longueur = strlen(argv[1]);
	printf("\nlongueur = %d\n",longueur);
	// Pas de '.' au bon endroit
	if (argv[1][longueur-4] != '.') usage(argc,argv);
	// Extension du fichier d'entrée non valide
	if (!extension_valide(argv[1]+longueur-3)) usage(argc,argv);
	// Extension en sortie non valide
	if (!extension_valide(argv[2])) usage(argc,argv);
	
	// Si on arrive là tout est OK et on calcule les noms des entrées et des sorties
	// Variable globale stockant le nom du fichier d'entrée
	NOM_FIC_IN = argv[1];
	
	// Recopie dans NOM_BASE du nom de fichier
	// on va mettre un 0 en position longueur-4
	// et EXT_IN pointe sur longueur-3
	NOM_BASE = malloc((longueur+1)*sizeof(char)); if (!NOM_BASE) { fprintf(stderr,"cannot malloc NOM_BASE\n"); exit(2); }
	sprintf(NOM_BASE,"%s",argv[1]);
	NOM_BASE[longueur-4] = '\0';
	EXT_IN = NOM_BASE + longueur - 3 ;
	
	// Calcul des noms en sortie
	EXT_OUT = argv[2];
	NOM_FIC_OUT = malloc((longueur+1)*sizeof(char));  if (!NOM_FIC_OUT) { fprintf(stderr,"cannot malloc NOM_FIC_OUT\n"); exit(2); }
	sprintf(NOM_FIC_OUT,"%s.%s",NOM_BASE,EXT_OUT);
	NOM_FIC_sz = malloc(longueur*sizeof(char));  if (!NOM_FIC_sz) { fprintf(stderr,"cannot malloc NOM_FIC_sz\n"); exit(2); }
	sprintf(NOM_FIC_sz,"%s.sz",NOM_BASE);
	NOM_ECLATE = malloc((strlen(NOM_BASE)+20)*sizeof(char)); if (!NOM_ECLATE) { fprintf(stderr,"cannot malloc NOM_ECLATE\n"); exit(2); }
	sprintf(NOM_ECLATE,"%s.eclate",NOM_BASE);
	printf("NOM_ECLATE_= %s\n",NOM_ECLATE);
	NOM_TRI = malloc((strlen(NOM_BASE)+13)*sizeof(char)); if (!NOM_TRI) { fprintf(stderr,"cannot malloc NOM_ECLATE\n"); exit(2); }
	sprintf(NOM_TRI,"%s.trieclate",NOM_BASE);

	// Affichage pour debug
#ifdef DEBUG
	printf("NOM_FIC_IN = %s\n",NOM_FIC_IN);
	printf("NOM_FIC_OUT = %s\n",NOM_FIC_OUT);
	printf("EXT_IN = %s\n",EXT_IN);
	printf("EXT_OUT = %s\n",EXT_OUT);
	printf("NOM_BASE = %s\n",NOM_BASE);
	printf("NOM_FIC_sz = %s\n",NOM_FIC_sz);
	printf("NOM_ECLATE_= %s\n",NOM_ECLATE);
#endif
}

// Lit la taille de la matrice dans le fichier .sz
void lecture_tailles() {
	printf("Je li la taille\n");
	FILE *F = fopen(NOM_FIC_sz,"r");
	
	if (!F) { fprintf(stderr,"Cannot open %s\n",NOM_FIC_sz); exit(3); }
	fscanf(F,"%llu",&nb_vals);
	fscanf(F,"%llu",&nb_lignes);
	fscanf(F,"%llu",&nb_automates);
	nb_colonnes = nb_lignes;
	min_l = min_c = 0;
	max_l = max_c = min_l + nb_lignes-1;
	
	if(VERSION == 2)
	{
		if (fscanf(F,"%llu",&nb_colonnes) < 1) {printf("je sort\n");return ; }
	
		if (fscanf(F,"%llu%llu",&min_l,&min_c) < 1) return;
	}
	max_l = min_l + nb_lignes-1;
	max_c = min_c + nb_colonnes-1;
	// Allocation de stableaux pour les degrés
	
	deg_l = calloc(nb_lignes,sizeof(unsigned long long int));
	if (!deg_l) { fprintf(stderr,"Cannot calloc deg_l\n"); exit(41); }
	deg_c = calloc(nb_colonnes,sizeof(unsigned long long int));
	if (!deg_c) { fprintf(stderr,"Cannot calloc deg_c\n"); exit(42); }
	printf("passage ok\n");
	fclose(F);
}

// Eclate la matrice en mettant dans un fichier, une suite de triplets (l,c,val) ou (c,l,val)
// en fonction de ce que l'on veut en sortie
void eclate_matrice() {
	printf("Eclate Matrice \n");
	
	FILE *F_eclate = fopen(NOM_ECLATE,"w");  if (!F_eclate ) { fprintf(stderr,"Cannot open file %s\n",NOM_ECLATE); exit(2); }
	FILE *F_in = fopen(NOM_FIC_IN,"r");  if (!F_in) { fprintf(stderr,"Cannot open file %s\n",NOM_FIC_IN); exit(2); }
	
	unsigned long long int i,j;
	unsigned long long int N, ind1, ind2, lemin;
	unsigned long long int nbval;
	double proba;
	if (EXT_IN[0]=='R') { N = nb_lignes; lemin=min_l; } else { N = nb_colonnes; lemin = min_c; }
	for (i=0 ; i<N ; i++) {
		fscanf(F_in,"%llu",&ind1);
		if (ind1!=i+lemin) { 
			fprintf(stderr,"Input matrix error, index of line %llu in the file is %llu, should be %llu\n",i+1,ind1,i+lemin);
			exit(51);
		}
		fscanf(F_in,"%llu",&nbval);
		if (EXT_IN[0]=='R') deg_l[i] = nbval;
		if (EXT_IN[0]=='C') deg_c[i] = nbval;
		for (j=0 ; j<nbval ; j++) {
			fscanf(F_in,"%le",&proba);
			fscanf(F_in,"%llu",&ind2);
			if (EXT_IN[0]=='R') (deg_c[ind2-min_c])++;
			if (EXT_IN[0]=='C') (deg_l[ind2-min_l])++;
			if (EXT_IN[0]==EXT_OUT[0]) fprintf(F_eclate,"%llu %llu %le\n",ind1,ind2,proba);
				else fprintf(F_eclate,"%llu %llu %le\n",ind2,ind1,proba);
		}
	}

	fclose(F_in);
	fclose(F_eclate);
	
}

void affiche_degres() {
	unsigned long long int i;
	for ( i=0 ; i<nb_lignes   ; i++) printf("deg_l[%llu] = %llu\n",i+min_l,deg_l[i]);
	for ( i=0 ; i<nb_colonnes ; i++) printf("deg_c[%llu] = %llu\n",i+min_c,deg_c[i]);
}

void trie_matrice_eclatee() {
	pid_t pid;
	pid = fork();
	if (pid == -1) { perror("fork"); exit(4); }
	char ordre1[5];
	char ordre2[5];
	if (EXT_OUT[1]=='d') sprintf(ordre1,"+0nr"); else sprintf(ordre1,"+0n");
	if (EXT_OUT[2]=='d') sprintf(ordre2,"+1nr"); else sprintf(ordre2,"+1n");
	if (pid == 0) execlp("sort","sort",ordre1,"-0",ordre2,"-o",NOM_TRI,NOM_ECLATE,NULL);
	wait(NULL);
#ifndef DEBUG
	unlink(NOM_ECLATE);
#endif
}

struct transition {
	unsigned long long int ind;
	double proba;
};

void relire_matrice() {
	FILE *F_tri = fopen(NOM_TRI,"r"); if (F_tri == NULL) {perror("fopen nomtrie"); exit(2);}
	FILE *F_OUT = fopen(NOM_FIC_OUT,"w"); if (F_OUT == NULL) {perror("fopen nomout"); exit(2);}

	unsigned long long int N, deb1, i, j, nbval, *deg1;
	unsigned long long int ind1, ind1_lu, ind2;
	double proba;
	int ret;
	if (EXT_OUT[0]=='R') { N = nb_lignes;   deb1 = min_l; deg1 = deg_l; }
	if (EXT_OUT[0]=='C') { N = nb_colonnes; deb1 = min_c; deg1 = deg_c; }
	for (i=0 ; i<N ; i++) {
		if (EXT_OUT[1]=='d') ind1 = deb1 + N-i -1; else ind1 = deb1 + i;
		nbval = deg1[ind1-deb1];
		fprintf(F_OUT,"%llu %llu     ",ind1,nbval);
		for (j=0 ; j<nbval ; j++) {
			ret = fscanf(F_tri,"%llu%llu%le",&ind1_lu,&ind2,&proba);
			if (ret!=3) { fprintf(stderr,"Incorrect number of values in file the %lluth value of index %llu\n",j,ind1); exit(61); }
			if (ind1_lu!=ind1) { fprintf(stderr,"Incorrect value of ind1 : %llu should be %llu\n",ind1_lu,ind1); exit(62); }
			fprintf(F_OUT," %le %llu",proba,ind2);
		}
		fprintf(F_OUT,"\n");
	}
	fclose(F_tri);
	fclose(F_OUT);
#ifndef DEBUG
	unlink(NOM_TRI);
#endif
}

void liberer_memoire()
{
	if ( deg_c != NULL)
		free(deg_c);
	if ( deg_l != NULL)
		free(deg_l);
	if(NOM_BASE != NULL)
		free(NOM_BASE);
	if(NOM_FIC_OUT != NULL)
		free(NOM_FIC_OUT);
	if(NOM_FIC_sz != NULL)
		free(NOM_FIC_sz);
	if(NOM_ECLATE != NULL)
		free(NOM_ECLATE);
	if(NOM_TRI != NULL)
		free(NOM_TRI);
}

int main (int argc, char *argv[]) {
	struct timeval tv;
	double date_debut, date_fin;
	gettimeofday(&tv,NULL);
	date_debut = tv.tv_sec + tv.tv_usec*1e-6;
	calcul_noms(argc,argv);
	printf("%s \n",NOM_ECLATE);
	if (strcmp(EXT_IN,EXT_OUT)==0) { fprintf(stderr,"Nothing to do same input '%s' and output '%s' format\n",EXT_IN,EXT_OUT); exit(0); }
	printf("1. READ AND SPLIT: %s -> %s \n",NOM_FIC_IN,NOM_ECLATE);
	lecture_tailles();
	eclate_matrice();
#ifdef DEBUG2
	affiche_degres();
#endif
	printf("\tRows  : %llu  [%llu..%llu]\n",nb_lignes,min_l,max_l);
	printf("\tCols  : %llu  [%llu..%llu]\n",nb_colonnes,min_c,max_c);
	printf("\tVals  : %llu non-zero values\n",nb_vals);
	printf("\tAutom : %llu automaton(s)\n",nb_automates);
// Test s'il manque des indices de ligne ou de colonne au début ou à la fin
	if ((EXT_OUT[0]=='R') && (max_l-min_l < nb_lignes-1))   { fprintf(stderr,"%llu rows indices are missing: ABORT\n",nb_lignes-1   - (max_l-min_l)); exit(31); }
	if ((EXT_OUT[0]=='C') && (max_c-min_c < nb_colonnes-1)) { fprintf(stderr,"%llu cols indices are missing: ABORT\n",nb_colonnes-1 - (max_c-min_c)); exit(32); }
	gettimeofday(&tv,NULL); date_fin = tv.tv_sec + tv.tv_usec*1e-6;
	printf("   %.3lf secondes\n",date_fin-date_debut);

	printf("2. SORT: %s -> %s\n",NOM_ECLATE,NOM_TRI);
	trie_matrice_eclatee();
	gettimeofday(&tv,NULL); date_fin = tv.tv_sec + tv.tv_usec*1e-6;
	printf("   %.3lf secondes\n",date_fin-date_debut);

	printf("3. READ AND WRITE: %s -> %s\n",NOM_TRI,NOM_FIC_OUT);
	relire_matrice();
	gettimeofday(&tv,NULL); date_fin = tv.tv_sec + tv.tv_usec*1e-6;
	gettimeofday(&tv,NULL); date_fin = tv.tv_sec + tv.tv_usec*1e-6;
	printf("   %.3lf secondes\n",date_fin-date_debut);
	liberer_memoire();

	return 0;
}
