#include <iostream>
#include <cmath>
using namespace std;

#define NMAX 50
struct Point {
	int x, y;
	int code;
	bool deleted;
	static double distanceSquared( Point a, Point b ) {
		return pow( a.x - b.x, 2 ) + pow( a.y - b.y, 2 );
	}
} A[ NMAX ], B[ NMAX ];
double minDist;
int Na, Nb;
int over[ NMAX * NMAX ][ 2 ], K = 0;

void inputOffer( int &N, Point p[] ) {
	cin >> N;
	for( int i = 0; i < N; i++ ) {
		cin >> p[ i ].code
		    >> p[ i ].x 
		    >> p[ i ].y;
	}
}
void buildOver() {
	for( int a = 0; a < Na; a++ ) {
		for( int b = 0; b < Nb; b++ ) {
			if( Point::distanceSquared( A[ a ], B[ b ] ) <= minDist ) {
				over[ K ][ 0 ] = A[ a ].code;
				over[ K ][ 1 ] = B[ b ].code;
				K++;
			}
		}
	}
}
void printOver() {
	for( int i = 0; i < K; i++ ) {
		cout << over[ i ][ 0 ] << ' ' 
		     << over[ i ][ 1 ] << '\n';
	}
}
void removeInterfering() {
	for( int a = 0; a < Na; a++ ) {
		for( int b = 0; b < Nb; b++ ) {
			if( A[ a ].deleted or B[ b ].deleted ) {
				continue;
			}
			if( Point::distanceSquared( A[ a ], B[ b ] ) <= minDist ) {
				A[ a ].deleted = true;
				B[ b ].deleted = true;
			}
		}
	}	
}
void printOffer( int &N, Point p[] ) {
	for( int i = 0; i < N; i++ ) {
		if( !p[ i ].deleted ) {
			cout << p[ i ].code << ' '  
			     << p[ i ].x << ' ' 
			     << p[ i ].y << '\n' ;
		}
	}
}
int main() {
	cin >> minDist;
	minDist = pow( minDist, 2 ); //compare squares of distances
	inputOffer( Na, A );
	inputOffer( Nb, B );
	//Get all interfering points in over[] and output them
	buildOver();
	printOver();
	//Remove interfering points and output the offers
	removeInterfering();
	printOffer( Na, A );
	printOffer( Nb, B );
	return 0;
}
