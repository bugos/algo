#include <stdio.h>

using namespace std;

#define ROWS 8
unsigned long long int ncol = 1; //number of colourings, the answer

int main() {

    for (int i = 1; i <= ROWS; i++) {
        #define ROW_LENGTH  2*i-1
        for (int j = 1; j <= ROW_LENGTH; j++) {

            //4 cases
            if (j == 1) //first triangle takes all three colors
                ncol *= 3;

            else if (j == 2) //second triangle (special case, contacts with 3-color triangles)
                ncol += ncol / 3;

            else if (j & 1) //j odd (not 1)
                ncol *= 2;

            else /* if not (j & 1) */ //j even (not 2)
                //ncol += ncol / 2; WRONG
                //ncol += ncol / 4 + ncol / 4; WRONG
                //ncol += (1/3) * (1/2) * ncol + (2/3) * (1/4) * ncol;
                //same 2 color options: 1/3 of the time and then same color: 1/2 of the time
                //different 2 color options: 2/3 of the time and then same color: 1/4 of the time
                ncol += ncol / 3;


                printf("j=%d: %llu\n", j, ncol);
        }
        printf("**i=%d: %llu\n", i, ncol);
    }
    printf("%llu\n", ncol);
}
