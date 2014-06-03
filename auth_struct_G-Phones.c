// Developer Evangelos 'bugos' Mamalakis
// A showcase of basic file and string handling in C.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// STRINGIFY allows us to use a macro as string width in scanf.
// http://stackoverflow.com/questions/3301294/scanf-field-lengths-using-a-variable-macro-c-c
#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x
#define BOOL char
#define TRUE 1
#define FALSE 0
#define REPEAT_FOR( times ) int i; for ( i = 0; i < times; i++ )

#define SUBSCRIBER_LIST "numbers.txt"
#define PHONE_LENGTH 10
typedef char PhoneString[ PHONE_LENGTH + 1 ];
typedef long long int PhoneInt; // The problem description specifies that the caller is of type long long int.

int NSubscribers; // FEAT: NSubscribers is not needed as we cand detect EOF of SUBSCRIBER_LIST.

FILE *openFile( char *filename, char *mode ) {
    FILE *file = fopen( filename, mode );
    if ( file == NULL ) {
        printf( "ERROR: Unable to open %s as %s", filename, mode );
        exit( 1 );
    }
    return file;
}
int inputPhoneString( PhoneString number ) {
    do { // Read exactly PHONE_LENGTH digits discarding the remaining chars to a newline.
        scanf( "%" STRINGIFY( PHONE_LENGTH ) "s" "%*[^\n]", number );
    } while ( 0 != strcmp( "0", number ) && PHONE_LENGTH != strlen( number ) );
}
void inputPhoneInt( PhoneInt *number ) {
    do { // Read exactly PHONE_LENGTH digits discarding the remaining chars to a newline.
        scanf( "%" STRINGIFY( PHONE_LENGTH ) "lld" "%*[^\n]", number );
    } while ( *number < pow( 10, PHONE_LENGTH - 1 ) );
}
void inputSubscribersToFile() {
    FILE *subscriberList = openFile( SUBSCRIBER_LIST, "w" );

    printf( "Insert the number of the subscribers: " );
    scanf( "%d", &NSubscribers );

    PhoneString newSubscriber;
    REPEAT_FOR( NSubscribers ) {
        printf( "Insert a valid subscriber number: " );
        inputPhoneString( newSubscriber ); // BUG: 0 passes from here.
        fprintf( subscriberList, "%s\n", newSubscriber );
        remove( newSubscriber ); // Clean up old data.
    }

    fclose( subscriberList );
}
BOOL isSubscriber( PhoneString number ) {
    FILE *subscriberList = openFile( SUBSCRIBER_LIST, "r" );
    PhoneString subscriber;
    BOOL found = FALSE;
    REPEAT_FOR( NSubscribers ) {
        fscanf( subscriberList, "%s\n", subscriber );
        if ( 0 == strcmp( subscriber, number ) ) {
            found = TRUE;
            break;
        }
    }

    fclose( subscriberList );
    return found;
}
void inputCallsToFiles() {
    FILE *subscriberCallLog;
    PhoneString recipient;
    PhoneInt caller;
    while( TRUE ) {
        printf( "Insert the recipient's number: " );
        inputPhoneString( recipient );
        if ( 0 == strcmp( "0", recipient ) ) {
            break; // Stop when we get 0 as a recipient.
        }
        if ( !isSubscriber( recipient ) ) {
            printf( "The number you entered does not belong to a subscriber. Try again.\n" );
            continue;
        }

        printf( "Insert the callers number: " );
        inputPhoneInt( &caller );

        subscriberCallLog = openFile( recipient, "a" );
        fprintf( subscriberCallLog, "%lld\n", caller );

        fclose( subscriberCallLog );
    }
}
void outputSubscriberCalls( char *subscriber ) {
    FILE *subscriberCallLog = openFile( subscriber, "r" );
    PhoneInt caller;
    while ( !feof( subscriberCallLog ) ) {
        int readArgs = fscanf( subscriberCallLog, "%lld", &caller );
        if ( 1 != readArgs ) { // fscanf reached eof.
            continue;
        }
        printf( "%lld\n", caller );
    }

    fclose( subscriberCallLog );
}
void outputCalls() {
    FILE *subscriberList = openFile( "numbers.txt", "r" );
    PhoneString subscriber;
    REPEAT_FOR( NSubscribers ) {
        fscanf( subscriberList, "%s\n", subscriber );

        printf( "Subscriber %s received the following calls:\n", subscriber );
        outputSubscriberCalls( subscriber );
    }

    fclose( subscriberList );
}
int main() {
    inputSubscribersToFile();
    inputCallsToFiles();

    outputCalls();
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
