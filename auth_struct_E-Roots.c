#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#define PRECISION 1e-8
#define FAILURE -1

double f( double x ) {
    return pow( x, 3);
    return pow( x, 3) - pow( x, 2 ) - x;
}
double fPrime( double x ) {
    return 3 * pow( x, 2 );
    return 3 * pow( x, 2 ) - 2 * x - 1;
}
double nroot( double x, int *depth, int maxDepth, double ( *f )(), double ( *fPrime )() ) {
    ++*depth; // We return depth + 1 so we know when we have reached maxDepth.
    if ( maxDepth < *depth ) {
        return x;
    }
    double fx = f( x );
    double fPrimex = fPrime( x );
    if ( 0 == fPrimex ) {
        *depth = FAILURE;
        return x;
    }
    double nextx = x - fx / fPrimex;
    if ( fabs( nextx - x ) < PRECISION ) {
        return x;
    }
    return nroot( nextx, depth, maxDepth, f, fPrime );
}
double froot( double x, double prevx, int *depth, int maxDepth, double ( *f )() ) {
    ++*depth; // We return depth + 1 so we know when we have reached maxDepth.
    if ( maxDepth < *depth ) {
        return x;
    }
    double fx = f( x );
    double prevfx = f( prevx );
    if ( 0 == fx - prevfx ) {
        *depth = FAILURE;
        return x;
    }
    double nextx = x - (x - prevx) * fx / ( fx - prevfx );
    if ( fabs( nextx - x ) < PRECISION ) {
        return x;
    }
    return froot( nextx, x, depth, maxDepth, f );
}
void printResult( char *name, double result, int depth, int maxDepth ) {
    printf( "%s()", name);
    if ( depth > maxDepth ) {
        depth = depth - 1;
        printf( " reached maxDepth and" );
    }
    if ( depth == FAILURE ) {
        printf( " failed with division by 0 and" );
    }
    printf( " returned %f after %d calls.\n", result, depth );
}
int main(void) {
    int maxDepth;
    printf( "Insert the maximum depth: " );
    scanf( "%d", &maxDepth );

    srand( time( NULL ) );
    const double random = rand();

    int nDepth = 0;
    double nResult = nroot( random, &nDepth, maxDepth, f, fPrime );

    int fDepth = 0;
    double fResult = froot( random, random * 2, &fDepth, maxDepth, f );

    printResult( "nRoot", nResult, nDepth, maxDepth );
    printResult( "fRoot", fResult, fDepth, maxDepth );
    return 0;
}
/*
Να γραφεί το πρόγραμμα στο οποίο να ορίζεται η συνάρτηση nroot(…) η οποία, μέσα από μια αναδρομική (recursive)  διαδικασία, υπολογίζει μια πραγματική ρίζα της εξίσωσης f(x)=0 προσεγγίζοντας την με την αναδρομική σχέση  xi+1=xi-f(xi)/f’(xi). Από τη σχέση αυτή παράγεται μία ακολουθία τιμών η οποία, κάτω από ορισμένες προϋποθέσεις, συγκλίνει προς μια πραγματική ρίζα της f(x)=0. Σε διαφορετική περίπτωση η ακολουθία αποκλίνει ή οι τιμές της παλινδρομούν. Ως αρχική τιμή για το x0 δίνεται ένας τυχαίος αριθμός.
Στο ίδιο πρόγραμμα να οριστεί και η συνάρτηση froot(…) η οποία να υπολογίζει, μέσα από μια αναδρομική (recursive) διαδικασία, μια πραγματική ρίζα της εξίσωσης f(x)=0 προσεγγίζοντας την με την αναδρομική σχέση xi+1 = xi-(xi-xi-1)f(xi)/(f(xi-f(xi-1)). Από τη σχέση αυτή παράγεται μία ακολουθία τιμών η οποία, κάτω από ορισμένες προϋποθέσεις, συγκλίνει προς μια πραγματική ρίζα της f(x)=0. Σε διαφορετική περίπτωση η ακολουθία αποκλίνει ή οι τιμές της παλινδρομούν.
Και για τις δύο συναρτήσεις η διαδικασία σταματά όταν θα ισχύει η σχέση |xi+1-xi |<e, όπου e ένας πολύ μικρός θετικός αριθμός που δηλώνει την ακρίβεια της μεθόδου. Ως αρχικές τιμές για το  x0 και  το x1 δίνονται τυχαίοι αριθμοί.
Για να συγκριθούν οι ταχύτητες σύγκλησης προς τη ρίζα των δύο συναρτήσεων, η συνάρτηση main του προγράμματος να τις καλεί για να υπολογίσουν, η κάθε μια χωριστά, τη ρίζα μιας εξίσωσης. Η main να τυπώνει τη ρίζα και τον αριθμό των επαναλήψεων που έκανε η κάθε συνάρτηση για να την προσεγγίσει με την ίδια ακρίβεια e. Η main να τυπώνει ακόμη και τα κατάλληλα μηνύματα στις περιπτώσεις που, για τη συγκεκριμένη εξίσωση, δε μπορεί να εφαρμοστεί ο αλγόριθμός ή η αντίστοιχη ακολουθία δε συγκλείνει προς τη ρίζα.
Να μη χρησιμοποιηθούν πουθενά γενικές μεταβλητές.
Οι συναρτήσεις να μη διαβάζουν δεδομένα και να μην εκτυπώνουν μηνύματα ή αποτελέσματα
Επειδή  στην περίπτωση πού δε μπορεί να υπολογιστεί μια πραγματική ρίζα από τις πιο πάνω αναδρομικές σχέσεις η διαδικασία θα συνεχίζεται επ’ άπειρο, να διαβάζεται ένας μέγιστος αριθμός επαναλήψεων για την εφαρμογή του κάθε αλγόριθμου.
Η έκφραση f’(xi) είναι η παράγωγος της f(x) για x ίσον με xi
*/
