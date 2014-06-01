// Developer Evangelos 'bugos' Mamalakis
// A non-optimal solution to the travelling salesman problem.
// Trying each city as origin, visit the nearest neighbor each time until we have visited all of them. 
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <limits.h>

#define FALSE 0
#define TRUE  1
#define BOOL char
#define FOR0( var, limit ) int var; for( var = 0; var < limit; var++ )

#define MAXN 50
#define UNDEFINED -1
int NCities, NRoads;
int distance[ MAXN ][ MAXN ];
BOOL visited[ MAXN ];
int minDistanceTravelled = INT_MAX, minOrigin;
int Route[ MAXN ], NStops = 0;

void inputGraph() {
    scanf( "%d %d", &NCities, &NRoads );
    assert( NCities < MAXN
         && NRoads  < pow( MAXN, 2 ) );
    assert( NRoads == NCities * (NCities - 1) / 2 ) // Complete Graph
    FOR0( i, NRoads ) {
        static int city1, city2, length;
        scanf( "%d %d %d", &city1, &city2, &length );
        assert( city1 < NCities
             && city2 < NCities
             && length > 0 );
        distance[ city1 ][ city2 ] = length;
        distance[ city2 ][ city1 ] = length;
    }
    FOR0( sameOriginDestination, NCities ) {
        distance[ sameOriginDestination ][ sameOriginDestination ] = 0;
    }
}
int findNearestNeighbour( int origin ) {
    int nearest = UNDEFINED;
    FOR0( neighbor, NCities ) {
        if ( !visited[ neighbor ] &&
           ( nearest == UNDEFINED || distance[ origin ][ neighbor ] < distance[ origin ][ nearest ] ) ) {
            nearest = neighbor;
        }
    }
    return nearest;
}
int travelFrom( int origin, BOOL saveRoute ) {
    FOR0( city, NCities ) {
        visited[ city ] = FALSE;
    }
    int current = origin, nearest = origin, distanceTravelled = 0;
    while( nearest != UNDEFINED ) {
        visited[ nearest ] = TRUE;
        distanceTravelled += distance[ current ][ nearest ];
        if ( saveRoute ) {
            Route[ NStops++ ] = nearest;
        }
        current = nearest;
        nearest = findNearestNeighbour( current );
    }
    return distanceTravelled;
}
void findMinDistanceTravelled() {
    FOR0( origin, NCities ) {
        int distanceTravelled = travelFrom( origin, FALSE );
        if( distanceTravelled < minDistanceTravelled ) {
            minDistanceTravelled = distanceTravelled;
            minOrigin = origin;
        }
    }
}
int main() {
    inputGraph();
    findMinDistanceTravelled();
    printf( "Travell %d from %d\n", minDistanceTravelled, minOrigin );
    travelFrom( minOrigin, TRUE );
    FOR0( stop, NStops ) {
        printf( "%d ", Route[ stop ] );
    }
    return 0;
}
/*
Σε ένα πλήρες γράφημα με N κορυφές η κάθε κορυφή του συνδέεται με όλες τις άλλες με ακμές  η κάθε μια από τις οποίες έχει ένα ορισμένο βάρος. Στο γράφημα αυτό ζητείται να βρεθεί η διαδρομή η οποία περνά, μια φορά, από όλες τις κορυφές του γραφήματος και το άθροισμα των βαρών των ακμών από τις οποίες περνά να είναι ελάχιστο (Πρόβλημα του περιοδεύοντας εμπόρου).
Για να βρεθεί μια τέτοια διαδρομή που να δίνει μια καλή λύση (όχι τη βέλτιστη) ακολουθείται ο εξής αλγόριθμος. Επιλέγεται μια κορυφή ως αρχική κορυφή εκκίνησης. Ως επόμενη κορυφή επιλέγεται αυτή που συνδέεται με την κορυφή εκκίνησης με το μικρότερο βάρος. Η κορυφή εκκίνησης διαγράφεται από το γράφημα και η διαδικασία επαναλαμβάνεται θέτοντας ως κορυφή εκκίνησης την επόμενη κορυφή. Η διαδικασία συνεχίζεται μέχρι να καλυφθούν όλες οι κορυφές του γραφήματος. Επειδή με την πιο πάνω διαδικασία η επιλογή της αρχικής κορυφής εκκίνησης επηρεάζει το τελικό αποτέλεσμα η διαδικασία εφαρμόζεται διαδοχικά χρησιμοποιώντας μια προς μία όλες τις κορυφές του γραφήματος ως αρχικές κορυφές εκκίνησης.
Να γραφεί το πρόγραμμα που υλοποιεί τον πιο πάνω αλγόριθμό.
Για την καταχώρηση των βαρών των ακμών του γραφήματος να ορίσετε τον πίνακας dis τύπου ΝxΝ ο οποίος, ως τιμή για το στοιχείο dis[i][j], να έχει το βάρος της ακμής που συνδέει την κορυφή i με την κορυφή j.
*/
