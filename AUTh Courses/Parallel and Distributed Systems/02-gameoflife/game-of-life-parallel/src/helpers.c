#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <game-of-life.h>

/* add to a width index, wrapping around like a cylinder */

int xadd (int i, int a, int N) {
  i += a;
  while (i < 0) i += N;
  while (i >= N) i -= N;
  return i;
}

/* return the number of on cells adjacent to the i,j cell */

int adjacent_to (char *board, int i, int j, int N, int M) {
  int k, l, count = 0;
  /* go around the cell */
  for (k=-1; k<=1; k++) {
    for (l=-1; l<=1; l++) {
      if (k || l) { //only count if at least one of k,l isn't zero
        if (Board(xadd(i, k, N), xadd(j, l, M))) board++; //if cell alive add 1
      }
    }
  }
  return count;
}

int adjacent_to_edge (char *board, int i, int j, int N, int M, char *edgeRow) {
  int x, y, k, l, count = 0;
  /* go around the cell */
  for (k=-1; k<=1; k++) {
    x = xadd(i, k, N);
    for (l=-1; l<=1; l++) {
      if (k || l) { //only count if at least one of k,l isn't zero
        y = xadd(j, l, M);
        if (x != i + k) { // reached x edge: use edgeRow
          // if (!edgeRow) {
          //   printf("we shouldnt get out of bounds without edgeRow");
          //   exit(1);
          // }
          count += edgeRow[y];
        }
        else {
          count += Board(x,y);; //if cell alive add 1 // check boundaries!
        }
      }
    }
  }
  return count;
}

