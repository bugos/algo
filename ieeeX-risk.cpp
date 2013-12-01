#include <iostream>
#include <algorithm>
using namespace std;

#define MIN3(a, b, c) min( a, min(b, c) ) 

#define MAXN 1000
int N, M, A[MAXN][MAXN], J[MAXN], totalRisk;

void pr() {
	cout << '\n';
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			cout << A[i][j] << ' ';
		}
		cout << '\n';
	}
}

int main() {

	//input
	cin >> N >> M;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			cin >> A[i][j];
		}
	}
	

	//trasform A to hold the min risk to reach each cell starting from the top
	//starting from the top is equivalent to starting from the bottom.
	for (int i = 1; i < N; i++) {
		#define iBelow i-1
		
		// j = 0: we are at the left-most cell
		A[i][0] += min (A[iBelow][0], A[iBelow][1]);
	
		//general case
		for (int j = 1; j <= M - 2; j ++) {
			A[i][j] += MIN3 (A[iBelow][j-1], A[iBelow][j], A[iBelow][j+1]);
		}
		
		//j = M - 1: we are at the right-most cell
		A[i][M-1] += min (A[iBelow][M-2], A[iBelow][M-1]);
		
	}
	
	
	//find the beginning 
	int jstart = 0; //we will start from j with minimum risk preferring the left one.
	for (int j = 1; j < M; j++) {
		if (A[N-1][j] < A[N-1][jstart]) jstart = j;
	}
	totalRisk = A[N-1][jstart];
	
	
	//go through the route saving the j's
	int cur = jstart;
	J[N-1] = cur;
	for (int i = N - 1; i > 0; i--) {		
		#define iAbv i-1
		//find next j
		if (cur == 0) {// we are at the left-most cell
			if (A[iAbv][cur+1] < A[iAbv][cur] ) cur = cur + 1;
		}
		else if (cur == M-1) { // we are at the right-most cell
			if (A[iAbv][cur-1] <= A[iAbv][cur] ) cur = cur - 1;
		}
		else {//general case 
			if (A[iAbv][cur-1] <= A[iAbv][cur] && A[iAbv][cur-1] <= A[iAbv][cur+1]) cur = cur - 1;
			else if ( A[iAbv][cur+1] < A[iAbv][cur] ) cur = cur + 1;
		}
		J[iAbv] = cur;
	}	
	
	//output
	cout << "Minimum risk path = ";
	for (int i = 0; i < N; i++) {
		cout << '[' << i << ',' << J[i] << ']';
	}
	cout << '\n';
	
	cout << "Risks along the path = " << totalRisk;
    return 0;
}
