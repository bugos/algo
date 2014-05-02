// Developer Evangelos 'bugos' Mamalakis
// A non-optimal solution to the travelling salesman problem.
// Trying each city as origin, visit the nearest neighbor each time until all is visited. 

#include <iostream>
#include <assert.h>
#include <limits.h>
using namespace std;

#define MAXN 50
int NCities, NRoads;
int distance[ MAXN ][ MAXN ];
int visited[ MAXN ] = {};

void setDistance( int origin, int destination, int lenght) {
	distance[ origin ][ destination ] = lenght;
	distance[ destination ][ origin ] = lenght;
}
void inputGraph() {
	cin >> NCities, NRoads;
	assert( NCities < MAXN 
	     && NRoads  < pow( MAXN, 2 ) );
	for( int i = 0; i < NRoads; i++ ) { 
		static int origin, destination, length;
		cin >> origin
		    >> destination
		    >> length;
		assert( origin      < NCities - 1 
		     && destination < NCities - 1
		     && length > 0 );
		setDistance(origin, destination, length);
	}
}
int travelFrom( int origin ) {
	int current = origin;
	int distanceTravelled = 0;
	do {
		int nearest = -1;
		for( int neighbor = 0; neighbor < NCities; neighbor++ ) {
			if ( !visited[ neighbor ] and
		       ( neighbor == -1 or distance[ current ][ neighbor ] < distance[ current ][ nearest ] ) ) {
				nearest = neighbor;
			}
		}
		if( nearest != -1 ) { //still -1 when no more cities to visit.
			distanceTravelled += distance[ current ][ nearest ];
			current = nearest;
		}
	} while( nearest != -1 ); 
	return distanceTravelled;
}
void resetVisited() {
	for( int i = 0; i < NCities; i++ ) {
		visited[ i ] = false;
	}
}
void findOriginWithMinDistanceTravelled() {
	int minDistanceTravelled = INT_MAX;
	for( int origin = 0; origin < NCities; origin++ ) {
		distanceTravelled = travelFrom( origin );
		if( distanceTravelled < minDistanceTravelled ) {
			minDistanceTravelled = distanceTravelled;
			//todo keep route
		}
		resetVisited();
	}
}
int main() {
	inputGraph();
	findOriginWithMinDistanceTravelled();
	return 0;
}
