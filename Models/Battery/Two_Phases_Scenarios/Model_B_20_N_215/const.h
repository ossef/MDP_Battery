/*constantes a modifier pour chaque modele*/

#define NEt               3   /*nombre de composantes du vecteur d'etat : (X, H, S)*/
#define NbArrivals        4   /*nombre d'arrivée, arrivées Batch {0,1,2,3} */
#define NbService         2   /*Service Bernouilli {0,1} */
#define NbRelease         2   /*Battery release Brnouilli {0,1} */
#define NbPhase           2   /*Number of phases */
#define NbEvtsPossibles   NbArrivals*NbRelease*NbService*NbPhase  /*nombre de vecteurs d'arrivees possibles*/
#define Polynom			0   /* iteration du modele */
#define Epsilon1 0.0		
#define Epsilon2  0.0	
	
