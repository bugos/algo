/* Written by bugos 6/2013
Longest Increasing Subsequence [dp with tabulation = O(n^2)]
https://kth.kattis.scrool.se/problems/longincsubseq
*/

#include <stdio.h>
using namespace std;

#define MAXN 10000
int N, A[MAXN], prev[MAXN], lis[MAXN], posmax = 0, out[MAXN] ;


//Reads a new testcase and stores it in the A array. If EOF return false.
bool input() {
    int ret = scanf("%d", &N);

    if (ret == 1) {
        for(int i = 0; i<N; i++) {
            scanf("%d", &A[i]);
        }
        return true;
    }
    else {
        return false;
    }
}

//fills the prev[], lis[] arrays. 
//prev[i] points to the previous element of the lis ending at i. used to build the lis[] and later the out[] arrays.
//lis[i] stores the length of the lis ending at i (dp).
void build_prev_lis() {
    for (int i=0; i<N; i++) {

        //prev[i]
        prev[i] = -1;
        for (int j=0; j<i; j++) {
            if (A[j] < A[i] && lis[j] + 1 > lis[ prev[i] ]) {
                prev[i] = j;
            }
        }

        //lis[i]
        lis[i] = 1;
        if ( prev[i]!=-1 ) {
            lis[i] = lis[ prev[i] ] + 1;
        } //else (when prev[i]==-1) lis[i] remains 1
    }
}

//finds max value in the lis[] array
void find_posmax() {
    posmax = 0;
    for (int i=1; i<N; i++) {
        if (lis[i] > lis[posmax]) {
                posmax = i;
        }
    }
}

//fills the out[] array with the indexes of the elements of the lis
void build_out() {
    int i = posmax;
    #define k lis[i]-1 //counter

    do {
        out[k] = i;
        i = prev[i];
    } while (i != -1);
}

//print the length of the lis and the out[] array
void output() {
    #define lis_length lis[posmax]

    printf("%d\n", lis_length);

    for (int i = 0; i<lis_length; i++) {
        printf("%d ", out[i]);
    }
    printf("\n");
}


int main() {
    freopen ("longincsubseq.in","r",stdin);

    while ( input() == true ) { 
        build_prev_lis();
        find_posmax();
        build_out();
        output();
    }

}

