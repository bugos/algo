// Developer Evangelos 'bugos' Mamalakis
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#define FOR0( var, limit ) int var; for( var = 0; var <  limit; var++ )
#define FOR1( var, limit ) int var; for( var = 1; var <= limit; var++ )
#define NMAX 50
struct Point {
	int x, y;
	int code;
	bool deleted;
} A[ NMAX ], B[ NMAX ];
double minDist;
int Na, Nb;
int over[ NMAX * NMAX ][ 2 ], K = 0;

double pointDistanceSquared( struct Point a, struct Point b ) {
	printf( "d%lf: %d %d\n" ,pow( a.x - b.x, 2 ) + pow( a.y - b.y, 2 ), a.code, b.code);
	return pow( a.x - b.x, 2 ) + pow( a.y - b.y, 2 );
}
void inputOffer( int *Npointer, struct Point p[] ) {
	scanf( "%d", Npointer );
	FOR0( i, *Npointer ) {
		scanf( "%d %d %d", &p[ i ].code, &p[ i ].x, &p[ i ].y );
	}
}
void buildOver() {
	FOR0( a, Na ) {
		FOR0( b, Nb ) {
			if( pointDistanceSquared( A[ a ], B[ b ] ) <= minDist ) {
				over[ K ][ 0 ] = A[ a ].code;
				over[ K ][ 1 ] = B[ b ].code;
				K++;
			}
		}
	}
}
void printOver() {
	printf( "There are %d interfering points: \n", K );
	FOR0( i, K ) {
		printf( "%d %d\n", over[ i ][ 0 ], over[ i ][ 1 ] );
	}
}
void removeInterfering() {
	FOR0( a, Na ) {
		FOR0( b, Nb ) {
			if( A[ a ].deleted || B[ b ].deleted ) {
				continue;
			}
			if( pointDistanceSquared( A[ a ], B[ b ] ) <= minDist ) {
				A[ a ].deleted = true;
				B[ b ].deleted = true;
			}
		}
	}	
}
void printOffer( const int N, const struct Point p[], const char *name ) {
	printf( "Offer %s after removing the interfering points: \n", name );
	FOR0( i, N ) {
		if( !p[ i ].deleted ) {
			printf( "%d %d %d\n", p[ i ].code, p[ i ].x, p[ i ].y );
		}
	}
}
int main() {
	scanf( "%lf", &minDist );
	minDist = pow( minDist, 2 ); //compare squares of distances
	inputOffer( &Na, A );
	inputOffer( &Nb, B );
	//Get all interfering points in over[] and output them
	buildOver();
	printOver();
	//Remove interfering points and output the offers
	removeInterfering();
	printOffer( Na, A, "A");
	printOffer( Nb, B, "B");
	return 0;
}
/*
Σε μια περιοχή πρόκειται να εγκατασταθούν σημεία ασύρματης πρόσβασης στο internet. Για το έργο έχουν υποβάλει προσφορές οι εταιρίες Α και Β σε κάθε μια από τις οποίες εμφανίζεται, ο αριθμός των σημείων που θα εγκαταστήσει η εταιρία, ένας κωδικός για κάθε σημείο και οι συντεταγμένες της θέσης του. Η κάθε εταιρία δε μπορεί να καταθέσει προσφορά για περισσότερα από 50 σημεία Επειδή οι εταιρίες χρησιμοποιούν διαφορετικά συστήματα πρέπει η απόσταση μεταξύ δύο σημείων που δεν ανήκουν στην ίδια εταιρία να είναι μεγαλύτερη από μια ελάχιστη αποδεκτή απόσταση.
Να γραφεί το πρόγραμμα το οποίο να επεξεργάζεται τα στοιχεία που κατέθεσαν οι δύο εταιρίες και να σχηματίζει τον πίνακα over, με δύο στήλες, ο οποίος να περιέχει του κωδικούς των σημείων για όλα τα ζεύγη για τα οποία υπάρχει επικάλυψη.  Για κάθε ζεύγος, στην πρώτη στήλη του πίνακα να εμφανίζεται ο κωδικός του σημείου που ανήκει στην εταιρία Α και στην δεύτερη ο κωδικός του σημείου που ανήκει στην εταιρία Β.
Στη συνέχεια το πρόγραμμα, αφού εκτυπώσει τον πίνακα over, για κάθε σημείο της προσφοράς της εταιρίας Α, να ελέγχει αν υπάρχει σημείο στην προσφορά της εταιρίας Β με το οποίο να υπάρχει επικάλυψη. Αν ναι τα δύο σημεία να διαγράφονται από τις προσφορές και ο έλεγχος να συνεχίζεται για τα υπόλοιπα σημεία. Τέλος το πρόγραμμα να εκτυπώνει τους κωδικούς και τις συντεταγμένες των σημείων που έχουν μείνει στις προσφορές και για τα οποία δεν υπάρχει επικάλυψη.  
*/
