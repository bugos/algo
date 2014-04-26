#include <iostream>
#include <cmath>
using namespace std;

#define NMAX 50
struct Point {
	int x, y;
	int code;
	int deleted;
	static int distance( Point a, Point b ) {
		int distance_squared = pow( a.x - b.x, 2 ) - pow( a.y - b.y, 2 );
		return sqrt( ceil( distance_squared ) );
	}
} A[ NMAX ], B[ NMAX ];
int minDist, Na, Nb;
int over[ NMAX * NMAX ][ 2 ], K;

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
			if( Point::distance( A[ a ], B[ b ] ) <= minDist ) {
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
			if( Point::distance( A[ a ], B[ b ] ) <= minDist ) {
				A[ a ].deleted = true;
				B[ b ].deleted = true;
			}
		}
	}	
}
void printOffer(int &n, Point p[] ) {
	for( int i = 0; i < n; i++ ) {
		if( p )
		cout << p[ i ].code << ' '  
		     << p[ i ].x << ' ' 
		     << p[ i ].y << '\n' ;
	}
}
int main() {
	//input
	cin >> minDist;
	inputOffer( Na, A );
	inputOffer( Nb, B );
	//phase 1
	buildOver();
	printOver();
	//phase 2
	removeInterfering();
	printOffer( Na, A );
	printOffer( Nb, B );
	return 0;
}
