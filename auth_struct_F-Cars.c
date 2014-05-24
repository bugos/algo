#include <stdio.h>
#include <stdlib.h>

#define FOR0( var, limit ) int var; for ( var = 0; var < limit; var++ )
#define NO_VIOLATION 0
const double CATEGORIES[] = { 1, 1.1, 1.2, 1.3 };

int *allocateInt( int length ) {
    int *memory = malloc( length * sizeof( int ) );
    if ( memory == NULL ) {
        exit( 0 );
    }
    return memory;
}
int **allocateIntPointer( int length ) {
    int **memory = malloc( length * sizeof( int * ) );
    if ( memory == NULL ) {
        exit( 0 );
    }
    return memory;
}
int **allocateInt2D( int rows, int *columns ) {
    int **memory = allocateIntPointer( rows );
    FOR0( row, rows ) {
        memory[ row ] = allocateInt( columns[ row ] );
    }
    return memory;
}
int *inputInt( int length ) {
    int *array = allocateInt( length );
    FOR0( i, length ) {
        scanf( "%d", &array[ i ] );
    }
    return array;
}
int **inputInt2D( int rows, int *columns ) {
    int **array = allocateInt2D( rows, columns );
    FOR0( row, rows ) {
        FOR0( column, columns[ row ] ) {
            scanf( "%d", &array[ row ][ column ] );
        }
    }
    return array;
}
int findPenalty( int speed, int limit ) {
    FOR0( i, sizeof( CATEGORIES ) / sizeof( int ) ) {
        if ( speed <= CATEGORIES[ i ] * limit ) {
            return i;
        }
    }
}
int **penalty( int **speed, int *limit, int NCameras, int *NCars ) {
    int **violation = allocateInt2D( NCameras, NCars );
    FOR0( camera, NCameras ) {
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
                printf( "Car %2d was driving over speed limit at %3d km/h and commited a type %d violation.\n",
                    number[ camera ][ car ],
                    speed[ camera ][ car ],
                    violation[ camera ][ car ] );
            }
        }
    }
}
int main(void) {
    int NCameras;
    scanf( "%d", &NCameras );
    int *limit = inputInt( NCameras );
    int *NCars = inputInt( NCameras );
    int **speed = inputInt2D( NCameras, NCars );
    int **number = inputInt2D( NCameras, NCars );

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
