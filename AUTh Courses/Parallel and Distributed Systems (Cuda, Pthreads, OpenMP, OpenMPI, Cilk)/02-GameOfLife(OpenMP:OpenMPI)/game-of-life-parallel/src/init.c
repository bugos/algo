#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "omp.h"


#include <game-of-life.h>

/* set everthing to zero */

void initialize_board (char *board, int N, int M) {
  int i, j;
  
  #pragma omp parallel for private(i,j) collapse(2)
  for (i=0; i<N; i++)
    for (j=0; j<M; j++) 
      Board(i,j) = 0;
}

/* generate random table */

void generate_table (char *board, int N, int M, float threshold, int rank) {
  int i, j;
  unsigned myseed;
  #pragma omp parallel private(myseed)
  {
    myseed = time(NULL) + 17 * omp_get_thread_num() + rank;
    #pragma omp for private(j)
    for (i=0; i<N; i++) {
      for (j=0; j<M; j++) {
        Board(i,j) = (float)rand_r(&myseed) < threshold * (float)RAND_MAX;  
      }
    }
  }
}

