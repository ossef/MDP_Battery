#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "const.h"
#define DEBUG 0
#define MAXETATS 100000

void afficherEtats(int t1[MAXETATS],int t2[MAXETATS],int t3[MAXETATS],int t4[MAXETATS],int n)
{
	printf("[\n");
		for(int i=0;i<n;i++)
		printf("%d  %d  %d  %d \n",t1[i],t2[i],t3[i],t4[i]);
	printf("] \n");
}

void TrierEtats_Cd(int t1[MAXETATS],int t2[MAXETATS],int t3[MAXETATS],int t4[MAXETATS],int n) //Tri : (X1,Y1)<(X2,Y2) si X1<X2 ou que X1=X2 et Y1>Y2
{
 //Les états sont déjà trié, mais il reste à extraire l'etat (0, 0, 0) et le mettre à la fin
 int i,j,tmp1, index;
 int min = t3[0];

 //Recherche de l'heure minimale 
 for(i=1;i<n;i++)
 {
	if(t3[i] <min)
		min = t3[i];
 }
 for(i=0;i<n;i++)
 {
	if(  (t2[i] == 0) && (t3[i] == min) && (t4[i] == 0) )
	{
		index = i;
		break;
	}
 }
 if(index < n-1)
 {
	tmp1 = t1[index];

	//Décalage à gauche
	for (j = index; j < n - 1; j++) 
	{
		t1[j] = t1[j + 1];
		t2[j] = t2[j + 1];
		t3[j] = t3[j + 1];
		t4[j] = t4[j + 1];
	}

	// Placer l'élément (0, min, 0) à la fin
	t1[n-1] = tmp1;
	t2[n-1] = 0;
	t3[n-1] = min;
	t4[n-1] = 0;
 }
}

int Rechercher(int I[MAXETATS],int e,int n)
{
for(int i=0;i<n;i++)
	 {
		if (I[i] == e)
		return i; 	 
     }		
	 return -1;
}

void OrdonnerMatrice_Rii(FILE *frii,int I[MAXETATS],char s[100],char rii[200],int n)
{
FILE *friiout;
char  riiout[200];
int i,j,c;
int t=0;

strcpy(riiout,s);
strcat(riiout,"-reordre.Rii");

friiout = fopen(riiout,"w");
if(friiout == NULL)
 exit(2);


int iter,degre,e,k,arret;
long double proba;
c= 0;

while(c <n)
 {
	for(i=0;i<n;i++)
	 {
		fscanf(frii,"%d %d",&iter,&degre);
		if(iter == I[c])
		 {
			  fprintf(friiout,"%12d %12d ",t++,degre);
			  for(j=0;j<degre;j++)
			  {
				fscanf(frii,"%Le %d",&proba,&e);
				fprintf(friiout,"% .25LE%12d",proba,Rechercher(I,e,n)); // recherche le nouvel emplacement de l'etats
			  }
				fprintf(friiout,"\n");
			c++;	
		 }
		 else
		 {
			   for(j=0;j<degre;j++)
				fscanf(frii,"%Le %d",&proba,&e);
		 }
	 }
	 frii=fopen(rii,"r");
 }
 
 fclose(friiout);	
}


int main(int argc, char *argv[])
{
 FILE *fcd,*fsz,*fpi,*friiout,*fszout,*fcdout,*frii;
 char sz[100], cd[100], rii[100],szout[100], cdout[100], city[100], path[200];
 int I[MAXETATS];
 int X[MAXETATS];
 int H[MAXETATS];
 int S[MAXETATS];
 long double proba;
 int deadline;
 int i,n1,n2,n,seuil,Buffer;

 
 if(argc != 3)
 {
 	printf(" Erreur passez en parametre le nom du model \n <Exemple d'usage > ./reordonner model50 Chicago \n");
 	exit(1);
 }
	 
	 strcpy(city,argv[2]);
	 snprintf(path, sizeof(path), "%s/%s", city, argv[1]);

	 strcpy(sz,path); 
	 strcpy(cd,path);
	 strcpy(rii,path);
	 strcpy(szout,path); 
	 strcpy(cdout,path);

  	 strcat(sz,".sz"); // On concatène chaine2 dans chaine1
  	 strcat(cd,".cd");
  	 strcat(rii,".Rii");
  	 strcat(szout,"-reordre.sz"); 
	 strcat(cdout,"-reordre.cd");




 fsz = fopen(sz,"r");
 if(fsz == NULL)
 exit(2);


 fcd = fopen(cd,"r");
 if(fcd == NULL)
 exit(2);


 frii = fopen(rii,"r");
 if(frii == NULL)
 exit(2);


 fszout = fopen(szout,"w");
 if(fszout == NULL)
 exit(2);

 
 fcdout = fopen(cdout,"w");
 if(fcdout == NULL)
 exit(2);


/*-------------------- Lecture du fichier ".sz" et ecriture dans ".reorder.sz" --------------*/

 fscanf(fsz,"%d",&n1);
 fscanf(fsz,"%d",&n);
 fscanf(fsz,"%d",&n2);
 fscanf(fsz,"%d",&seuil);
 fscanf(fsz,"%d",&Buffer);
 fprintf(fszout,"%12d \n%12d \n%12d \n%12d %12d \n",n1,n,n2,seuil,Buffer);
 
 
/*-------------------- Lecture du fichier ".cd" -----------------*/

for(i=0;i<n;i++)
 {
	fscanf(fcd,"%d %d %d %d",&I[i],&X[i],&H[i],&S[i]);
 }


if(DEBUG == 1)
{
  printf("Avant trie :\n");
  afficherEtats(I,X,H,S,n);
}

/*------------------ Tri des états !! ----------------------------*/

TrierEtats_Cd(I,X,H,S,n);


if(DEBUG == 1)
{  
  printf("\n Aprés trie :\n");
  afficherEtats(I,X,H,S,n);
}

/*-----------------Ecriture du nouveau fichier reorder.cd .........*/

  for(i=0;i<n;i++)
	fprintf(fcdout,"%12d %12d %12d %12d \n",i,X[i],H[i],S[i]);
	

/*---------------- Re ordonner et ecriture de la nouvel matrice ".reorder.Rii" !! -----*/
 
  OrdonnerMatrice_Rii(frii,I,path,rii,n); 

	fclose(fsz);
	fclose(fcd);
	fclose(frii);
	fclose(fszout);
	fclose(fcdout);
	printf("Matrix re-order done ! \n");
	
  return 0;
}
