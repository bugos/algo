// Developer Evangelos 'bugos' Mamalakis
#include <stdio.h>
#include <assert.h>

#define FOR0( var, limit ) int var; for( var = 0; var <  limit; var++ )
#define FOR1( var, limit ) int var; for( var = 1; var <= limit; var++ )

#define N_MOST_SUN 3
int KArreas, minSunDays;
int Ngood = 0;
int mostSun[ N_MOST_SUN ], mostSunW[ N_MOST_SUN ] = {}; //Priority Queue

void PQueueAdd( int value, int weigh ) {
    FOR0( i, N_MOST_SUN ) {
        if( weigh > mostSunW[ i ] ) { //found a spot
            //Move previous values one position to the right
            // overwriting the last value and starting from the end.
            int j; for( j = N_MOST_SUN - 1; j > i; j-- ) {
                mostSunW[ j ] = mostSunW[ j - 1 ];
                mostSun [ j ] = mostSun [ j - 1 ];
            }
            //Insert new values
            mostSunW[ i ] = weigh;
            mostSun [ i ] = value;
            break;
        }
    }
}
int main() {
    scanf( "%d %d", &KArreas, &minSunDays );
    int sunDays;
    FOR1( arrea, KArreas ) {
        scanf( "%d", &sunDays );
        if( sunDays < minSunDays ) {
        	printf( "Area %d did not have enough sun days.\n", arrea );
            continue;
        }
        ++Ngood;
        PQueueAdd( arrea, sunDays );
    }
    //Output
    printf( "There are %d areas with enough sun.\n", Ngood );
    printf( "Most Sunny Areas:\n" );
    FOR0( i, N_MOST_SUN ) {
        printf( "Area %d with %d sun days\n", mostSun[i], mostSunW[i] );
    }
    return 0;
}
