/* Written by bugOS on 8-7-2013
Problem Description: http://hellenico.gr/contest/task.php?name=trlights */

#include <stdio.h>
using namespace std;


int redCount, greenCount, dayRank;
char c;


int main() {
    //freopen("trlights.in" , "r", stdin );
    //freopen("trlights.out", "w", stdout);


    //input and counting
    #define N 10
    for (int i=1; i<=N; i++) {
        scanf("%c", &c);

        switch (c) {
            case 'r':
                redCount++;
                break;
            case 'g':
                greenCount++;
                break;
        }
    }


    #define VERY_BAD 1
    #define BAD 2
    #define GOOD 3
    #define VERY_GOOD 4
    if (redCount >= 5)
        dayRank = VERY_BAD;
    else if (greenCount == N)
        dayRank = VERY_GOOD;
    else if (redCount & 1) //is odd
        dayRank = BAD;
    else //redCount is even
        dayRank = GOOD;


    //output
    for (int i=1; i<=dayRank; i++) {
        printf("*");
    }
    printf("\n");

}
