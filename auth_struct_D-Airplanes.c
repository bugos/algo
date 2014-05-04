// Developer Evangelos 'bugos' Mamalakis
#include <stdio.h>
#include <math.h>

#define FOR( var, start, end ) int var; for( var = start; var < end; var++ )
#define FOR0( var, end ) FOR( var, 0, end )
#define BOOL char
#define TRUE 1
#define FALSE 0

#define DIMENSIONS 3
#define X coordinates[ 0 ]
#define Y coordinates[ 1 ]
#define Z coordinates[ 2 ]
#define RED 2
#define YELLOW 1
#define GREEN 0
#define SELF airspace[ 0 ]

#define MAXN 100

typedef struct Airplane {
	char id[ 10 ];
	int coordinates[ DIMENSIONS ];
} Airplane;
double airplaneDistance( Airplane a, Airplane b ) {
	double res = 0;
	FOR0( dim, DIMENSIONS ) {
		res += pow( a.coordinates[ dim ] - b.coordinates[ dim ], 2 );
	}
	res = sqrt( res );
	return res;
}
void inputAirplaneIds( int NPlanes, Airplane airspace[] ) {
	FOR0( plane, NPlanes ) {
		scanf( "%s", airspace[ plane ].id );
	}
}
void inputAirplane( Airplane *plane ) {
	scanf( "%d %d %d", &plane->X, &plane->Y, &plane->Z );
}
void inputAirspaceSnapshot( int NPlanes, Airplane airspace[] ) {
	FOR( plane, 1, NPlanes ) {
		inputAirplane( &airspace[ plane ] );
	}
}
int alarm( Airplane a, Airplane b, double safeDistance, double *distance ) {
	double dist = airplaneDistance( a, b );
	*distance = dist; // also return distance
	if( dist <= 0.75 * safeDistance )
		return RED;
	if( dist <= safeDistance )
		return YELLOW;
	return GREEN;
}
void checkForAlarms( int NPlanes, Airplane airspace[], double safeDistance ) {
	FOR( plane, 1, NPlanes ) {
		double dist;
		switch( alarm( SELF, airspace[ plane ], safeDistance, &dist ) ) {
			case GREEN:
				break;
			case YELLOW:
				printf( "Κίτρινος Συναγερμός: " );
				break;
			case RED:
				printf( "Κόκκινος Συναγερμός: " );
				break;
		}
		printf( "Plane %s is located %lf meters away\n", airspace[ plane ].id, dist );
	}
}
BOOL atAirport( Airplane plane ) {
	BOOL res = TRUE;
	FOR0( dim, DIMENSIONS ) {
		if ( plane.coordinates[ dim ] != 0 ) {
			res = FALSE;
			break;
		}
	}
	return res;
}
int main() {
	int NPlanes;
	scanf( "%d", &NPlanes );
	double safeDistance;
	scanf( "%lf", &safeDistance );
	Airplane airspace[ MAXN ];
	inputAirplaneIds( NPlanes, airspace );
	while( TRUE ) {
		inputAirplane( &SELF );
		if( atAirport( SELF ) ) {
			break;
		}
		inputAirspaceSnapshot( NPlanes, airspace );
		checkForAlarms( NPlanes, airspace, safeDistance );
	}
	return 0;
}
/*
Για να δοκιμαστεί ένα σύστημα αποφυγής των συγκρούσεων σε ένα από τα Ν αεροπλάνα ενός σμήνους εγκαθίσταται το σύστημα και το σμήνος εκτελεί μια δοκιμαστική πτήση. Το σύστημα με το οποίο εφοδιάζεται το αεροπλάνο ανιχνεύει, μέσω του ραντάρ, τις συντεταγμένες (x,y,z) των άλλων αεροπλάνων του σμήνους και μέσω του GPS τις δικές του συντεταγμένες. Το σύστημα υπολογίζει τις αποστάσεις από τα άλλα αεροπλάνα του σμήνους και αν κάποια από αυτές είναι μικρότερη από μια δοθείσα απόσταση ασφαλείας ενεργοποιεί ένα σήμα συναγερμού.
Να γραφεί το πρόγραμμα το οποίο να υλοποιεί το λογισμικό για τη λειτουργία του συστήματος. Στο πρόγραμμα να ορίζεται η συνάρτηση alarm(…) η οποία να δέχεται τις συντεταγμένες του αεροπλάνου που φέρει το σύστημα και τις συντεταγμένες ενός από τα άλλα αεροπλάνα του σμήνους. Η συνάρτηση να υπολογίζει την απόσταση των δυο αεροπλάνων και αν αυτή είναι μεγαλύτερη από την απόσταση ασφαλείας να επιστρέφει την τιμή 0. Αν η απόσταση είναι μικρότερη από την απόσταση ασφαλείας και μεγαλύτερη από τα ¾ της να επιστρέφει την τιμή 1. Σε κάθε άλλη περίπτωση να επιστρέφει την τιμή 2. Η συνάρτηση σε κάθε περίπτωση να επιστρέφει και την απόσταση των δυο αεροπλάνων.       
Το πρόγραμμα να διαβάζει την απόσταση ασφαλείας και, ως ταυτότητα, για κάθε αεροπλάνο ένα string από 10 το πολύ χαρακτήρες. Στη συνέχεια να ορίζεται μια ατέρμων ανακύκλωση σε κάθε επανάληψη της οποίας να διαβάζονται η συντεταγμένες του αεροπλάνου που φέρει το σύστημα και για κάθε ένα από τα άλλα αεροπλάνα, αφού διαβαστούν οι συντεταγμένες του, να καλείται η συνάρτηση alarm για να υπολογιστεί η απόσταση των αεροπλάνων και το είδος του σήματος συναγερμού. Αν η επιστρεφόμενη από τη συνάρτηση τιμή είναι 0 να εκτυπώνεται η ταυτότητα του αεροπλάνου και η απόστασή του. Αν η επιστρεφόμενη τιμή είναι 1 να εκτυπώνεται το μήνυμα «Κίτρινος Συναγερμός», η ταυτότητα του αεροπλάνου και η απόστασή του,  Αν η επιστρεφόμενη τιμή είναι 2 να εκτυπώνεται το μήνυμα «Κόκκινος Συναγερμός», η ταυτότητα του αεροπλάνου και η απόστασή του,
Η ανακύκλωση να τερματίζεται όταν προσγειωθεί το αεροπλάνο.
Σημείωση: Να μη χρησιμοποιηθούν πουθενά γενικές μεταβλητές.
Οι συντεταγμένες των αεροπλάνων ορίζονται σε ένα τοπικό καρτεσιανό σύστημα συντεταγμένων του οποίου η αρχή βρίσκεται επάνω στο διάδρομο του αεροδρομίου και ο άξονας των z είναι κατακόρυφος.
Η συνάρτηση alarm να μη διαβάζει δεδομένα και να μην εκτυπώνει αποτελέσματα.
*/
