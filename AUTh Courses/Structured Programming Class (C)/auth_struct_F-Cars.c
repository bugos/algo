#include <stdio.h>
#include <stdlib.h>

#define FOR0( var, limit ) int var; for ( var = 0; var < limit; var++ )
#define NO_VIOLATION 0
const double CATEGORY_LIMITS[] = { 1, 1.1, 1.2 };

void *allocate( int length, int size, const char* callerName ) {
#define allocate( a,b ) allocate( a,b, __FUNCTION__ )
    void *memory = malloc( length * size );
    if ( memory == NULL ) {
        printf( "%s() failed to allocate %d %d-sized parts of memory.\n", callerName, length, size );
        exit( 1 );
    }
    return memory;
}
int *inputInts( char *message, int length ) {
    int *array = allocate( length, sizeof *array );
    printf( message, length );
    FOR0( i, length ) {
        scanf( "%d", &array[ i ] );
    }
    return array;
}
int **inputInts2D( char *message, char *rowMessage, int rows, int *columns ) {
    int **array = allocate( rows, sizeof *array );
    printf( message, rows );
    FOR0( row, rows ) {
        array[ row ] = allocate( columns[ row ], sizeof **array );
        array[ row ] = inputInts( rowMessage, columns[ row ] );
    }
    return array;
}
int findPenalty( int speed, int limit ) {
    int NCategories = sizeof CATEGORY_LIMITS / sizeof( int );
    FOR0( category, NCategories ) {
        if ( speed <= CATEGORY_LIMITS[ category ] * limit ) {
            return category;
        }
    }
    return NCategories;
}
int **penalty( int **speed, int *limit, int NCameras, int *NCars ) {
    int **violation = allocate( NCameras, sizeof *violation );
    FOR0( camera, NCameras ) {
        violation[ camera ] = allocate( NCars[ camera ], sizeof **violation );
        FOR0( car, NCars[ camera ] ) {
            violation[ camera ][ car ] = findPenalty( speed[ camera ][ car ], limit[ camera ] );
        }
    }
    return violation;
}
void printViolators( int **number, int **speed, int **violation, int NCameras, int *NCars ) {
    FOR0( camera, NCameras ) {
        FOR0( car, NCars[ camera ] ) {
            if ( violation[ camera ][ car ] != NO_VIOLATION ) {
                printf( "Car #%04d was driving over speed limit at %3d km/h and commited a type %d violation.\n",
                    number[ camera ][ car ],
                    speed[ camera ][ car ],
                    violation[ camera ][ car ] );
            }
        }
    }
}
int main(void) {
    int NCameras;
    printf( "Enter the number of the cameras: " );
    scanf( "%d", &NCameras );
    int *limit = inputInts( "Enter the limits for %d cameras:\n", NCameras );
    int *NCars = inputInts( "Enter the number of cars for %d cameras:\n", NCameras );
    int **speed = inputInts2D(
        "Enter the speeds of the cars for %d cameras.\n",
        "Enter the speeds for camera %d:\n",
        NCameras, NCars
    );
    int **number = inputInts2D(
        "Enter the registration numbers of the cars for each camera.\n",
        "Enter the registration numbers for camera %d:\n",
        NCameras, NCars
    );

    int **violation = penalty( speed, limit, NCameras, NCars );

    printViolators( number, speed, violation, NCameras, NCars );
    return 0;
}
/*
Σε μια περιοχή έχουν εγκατασταθεί Ν κάμερες της τροχαίας με στόχο να ελεγχθούν οι παραβιάσεις του ορίου ταχύτητας. Όταν ένα αυτοκίνητο περάσει από το σημείο που βρίσκεται η κάμερα και παραβιάσει το όριο ταχύτητας, καταγράφεται ο αριθμός κυκλοφορίας και η ταχύτητα του αυτοκινήτου. Στη συνέχεια τα αυτοκίνητα που παραβίασαν το όριο ταχύτητας κατατάσσονται σε τρεις κατηγορίες.  Στην  πρώτη ανήκουν αυτά που η υπέρβαση ήταν μέχρι 10% πάνω από το όριο της ταχύτητας, στη δεύτερη αυτά που η υπέρβαση ήταν πάνω από το 10% και κάτω από το 20% του ορίου και στην τρίτη αυτά που η ταχύτητα τους υπερέβαινε το 20% του ορίου.
Να γραφεί το πρόγραμμα στο οποίο ορίζεται η συνάρτηση penalty (…) η οποία δέχεται ένα πίνακα δύο διαστάσεων ο οποίος περιέχει σε κάθε γραμμή τις ταχύτητες που κατέγραψε η αντίστοιχη κάμερα και επιστρέφει έναν πίνακα του ιδίου τύπου ο οποίος στις αντίστοιχες θέσεις περιέχει την κατηγορία της αντίστοιχης παράβασης
Το πρόγραμμα να διαβάζει τον αριθμό των καμερών, το όριο ταχύτητας που τέθηκε για κάθε κάμερα και τον αριθμό των αυτοκινήτων που παραβίασαν το όριο στη θέση της αντίστοιχης κάμερας. Στη συνέχεια το πρόγραμμα να σχηματίζει τους πίνακες speed  και number, δύο διαστάσεων και να διαβάζει και να καταχωρεί στον πρώτο τις ταχύτητες των αυτοκινήτων που παραβίασαν τα όρια ταχύτητας  και στον δεύτερο, ως έναν ακέραιο αριθμό, τον αριθμό κυκλοφορίας του αντίστοιχου αυτοκινήτου.
Το πρόγραμμα να καλεί τη συνάρτηση penalty για να υπολογίσει τον πίνακα με την κατηγορία της κάθε παράβασης και στη συνέχεια να εκτυπώνει για κάθε παραβάτη τον αριθμό του αυτοκινήτου την ταχύτητα και την κατηγορία της παράβασης.
Σημείωση: Οι μνήμη για τους πίνακες που θα χρησιμοποιηθούν να δεσμεύεται δυναμικά στο ελάχιστο απαιτούμενο μέγεθος σύμφωνα με τα δεδομένα που εισάγονται κάθε φόρά στο πρόγραμμα
Στους πίνακες speed  και number στην i γραμμή να αντιστοιχούν τα στοιχεία που συνέλεξε η υπ’ αριθμόν i κάμερα.
*/
