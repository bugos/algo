// Developer Evangelos 'bugos' Mamalakis
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#define FOR( var, start, end ) int var; for( var = start; var < end; var++ )
#define FOR0( var, end ) FOR( var, 0, end )
#define MAXN 100
#define DIMENSIONS 3
#define SELF airplanes[ 0 ]

enum alarmColor { GREEN, YELLOW, RED };
typedef struct Airplane {
	char id[ 10 + 1 ];
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
void inputAirplaneIds( int NPlanes, Airplane airplanes[] ) {
	FOR0( plane, NPlanes ) {
		scanf( "%s", airplanes[ plane ].id );
	}
}
void inputAirplanes( int start, int NPlanes, Airplane airplanes[] ) {
	FOR( plane, start, NPlanes ) {
		FOR0( dim, DIMENSIONS ) {
			scanf( "%d", &airplanes[ plane ].coordinates[ dim ] );
		}
	}
}
enum alarmColor alarm( Airplane a, Airplane b, double safeDistance, double *distance ) {
	*distance = airplaneDistance( a, b );
	if( *distance <= 0.75 * safeDistance )
		return RED;
	if( *distance <= safeDistance )
		return YELLOW;
	return GREEN;
}
void checkForAlarms( int NPlanes, Airplane airplanes[], double safeDistance ) {
	FOR( plane, 1, NPlanes ) {
		double distance;
		switch( alarm( SELF, airplanes[ plane ], safeDistance, &distance ) ) {
			case GREEN:
				break;
			case YELLOW:
				printf( "Κίτρινος Συναγερμός: " );
				break;
			case RED:
				printf( "Κόκκινος Συναγερμός: " );
				break;
		}
		printf( "Plane %s is located %lf meters away\n", airplanes[ plane ].id, distance );
	}
}
bool atAirport( Airplane plane ) {
	FOR0( dim, DIMENSIONS ) {
		if ( plane.coordinates[ dim ] != 0 ) {
			return false;
		}
	}
	return true;
}
int main() {
	int NPlanes;
	scanf( "%d", &NPlanes );
	double safeDistance;
	scanf( "%lf", &safeDistance );
	Airplane airplanes[ MAXN ];
	inputAirplaneIds( NPlanes, airplanes );
	while( true ) {
		inputAirplanes( 0, 1, airplanes );
		if( atAirport( SELF ) ) {
			break;
		}
		inputAirplanes( 1, NPlanes, airplanes );
		checkForAlarms( NPlanes, airplanes, safeDistance );
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
