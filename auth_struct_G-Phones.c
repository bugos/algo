#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPEAT_FOR( times ) int i; for ( i = 0; i < limit; i++ )
#define NUMBER_LENGTH 10

FILE *openFile( char *filename, char *mode ) {
    FILE *file = fopen( filename, mode );
    if ( file == NULL ) {
        exit( 0 );
    }
    return file;
}
int inputSubscribersToFile() {
    FILE *numbersList = openFile( "numbers.txt", "w" );

    int NSubscribers;
    printf( "Insert the number of the subscribers.\n" );
    scanf( "%d", &NSubscribers );

    printf( "Insert %d subscriber numbers.\n", NSubscribers );
    char number[ NUMBER_LENGTH ];
    REPEAT_FOR( NSubscribers ) {
        scanf( "%s", number );
        fprintf( numbersList, "%s\n", number );
        remove( number ); // Clean up old data.
    }
    fclose( numbersList );
    return NSubscribers;
}
void inputCallsToFiles() {
    FILE *recipientCallList;
    char recipient[ NUMBER_LENGTH ];
    long long int caller; // The specification requires a long long int.
    while( 1 ) {
        printf( "Insert the subscriber's number.\n" );
        scanf( "%s", recipient );
        if ( !strcmp( "0", recipient ) ) {
            break; // Stop when we get 0 as a recipient.
        }
        printf( "Insert the callers number.\n" );
        scanf( "%lld", &caller );

        recipientCallList = openFile( recipient, "a" );
        fprintf( recipientCallList, "%lld\n", caller );
        fclose( recipientCallList );
    }
}
void outputSubscriberCalls( char *subscriber ) {
    FILE *subscriberCallLog = openFile( subscriber, "r" );
    long long int caller;
    while ( !feof( subscriberCallLog ) ) {
        int readArgs = fscanf( subscriberCallLog, "%lld", &caller );

        if ( readArgs == 1 ) { // fscanf was successful
            printf( "%lld\n", caller );
        }
    }
    fclose( subscriberCallLog );
}
void outputCalls( int NSubscribers ) {
    FILE *numbersList = openFile( "numbers.txt", "r" );
    char subscriber[ NUMBER_LENGTH ];
    REPEAT_FOR( NSubscribers ) {
        fscanf( numbersList, "%s\n", subscriber );

        printf( "Subscriber %s received the following calls:\n", subscriber );
        outputSubscriberCalls( subscriber );
    }
    fclose( numbersList );
}
int main() {
    int NSubscribers = inputSubscribersToFile();
    inputCallsToFiles();
    outputCalls( NSubscribers );
    return 0;
}
/*
Για την καταχώρηση των εισερχομένων κλήσεων που έγιναν προς έναν αριθμό τηλεφώνου η εταιρία κρατά ένα αρχείο κλήσεων για κάθε συνδρομητή. Το όνομα του αρχείου ταυτίζεται με τον αριθμό κλήσης του συνδρομητή. Όταν ο συνδρομητής έχει μια εισερχόμενη κλήση αναζητείται το αντίστοιχο αρχείο το οποίο ανοίγει για να προστεθεί στο τέλος του ο αριθμός του τηλεφώνου από το οποίο έγινε η κλήση.
Να γραφεί το πρόγραμμα το οποίο αρχικά δημιουργεί ένα αρχείο με το όνομα numpers. Στη συνέχεια να διαβάζει τους αριθμούς κλήσης των συνδρομητών της εταιρίας και τους καταχωρεί με τη μορφή strings στο αρχείο. Αυτά τα strings θα είναι τα ονόματα των αρχείων που θα αντιστοιχούν σε κάθε συνδρομητή. Το πρόγραμμα να ορίζει μια ατέρμονα ανακύκλωση σε κάθε επανάληψη της οποίας να διαβάζεται ο αριθμός κλήσης ενός συνδρομητή και ο αριθμός του τηλεφώνου που πραγματοποίησε την κλήση. Στη συνέχεια να αναζητείται στο αρχείο numpers το όνομα του αρχείου που αντιστοιχεί στον συνδρομητή, να ανοίγει το αντίστοιχο αρχείο συνδεόμενο με ένα δυαδικό κανάλι και να καταχωρείται σε αυτό ο αριθμός του τηλεφώνου που τον κάλεσε. Η ανακύκλωση να σταματά αν ως αριθμός του συνδρομητή δοθεί το 0. Στην περίπτωση αυτή το πρόγραμμα να εκτυπώνει για κάθε συνδρομητή τους αριθμούς των τηλεφώνων που τον έχουν καλέσει.
Βοηθητικές παρατηρήσεις
Ο αριθμός του συνδρομητή να διαβάζεται και να καταχωρείται με τη μορφή string ενώ ο αριθμός του τηλεφώνου που τον καλεί με τη μορφή ακεραίου τύπου long.
Για την καταχώρηση και ανάγνωση των ονομάτων των αρχείων που αντιστοιχούν στους συνδρομητές να χρησιμοποιήσετε τις συναρτήσεις fscanf και fprintf.
Για τη σύγκριση των strings μπορείτε να χρησιμοποιήσετε έτοιμες συναρτήσεις που διαθέτει ο μεταγλωττιστής που χρησιμοποιείτε. (Η σχετική συνάρτηση στον μεταγλωττιστή της Borland είναι η strcmp και ορίζεται στο αρχείο string.h). 
*/
