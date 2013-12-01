//solves pe18 too
//bugos 9-9-13 agia galini
//Jukebox i am coming tonight!!!

#include <stdio.h>
#include <algorithm>
using namespace std;

#define ROWS 100
int T[ROWS + 1][ROWS + 1];

int feed_me() { //input
    freopen("pe67.in", "r", stdin);
    for (int i = 1; i <= ROWS; i++) {
        for (int j=1; j <= i; j++) {
            scanf("%d", &T[i][j]);
        }
    }
}

//start from the bottom and add the bigger
//of the two adjacent numbers below.
int dp_solve() {
    for (int i = ROWS - 1; i >= 1; i--) {
        for (int j=1; j <= i; j++) {
            T[i][j] += max(T[i+1][j], T[i+1][j+1]);
        }
    }
}

int main() {
    feed_me();
    dp_solve();
    printf("%d\n", T[1][1]);
    return 0; //win!
}
