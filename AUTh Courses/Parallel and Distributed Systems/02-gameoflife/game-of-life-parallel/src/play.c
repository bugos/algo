#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>
#include "omp.h"

#include <game-of-life.h>

int inline calculateCell(int cell, int a) {
    if (a == 2) return cell;
    if (a == 3) return 1; 
    return 0; // if a<2 or a>3
}

void inline calculateRow(char *board, char *newboard, int N, int M, int i, char *edgeRow) {
  int j;
  //#pragma omp parallel for
  for (j=0; j<M; j++) {
    int a = adjacent_to_edge(board, i, j, N, M, edgeRow);
    NewBoard(i,j) = calculateCell(Board(i,j), a);
  }
}

int inline modulus(int a, int b) {
  return (a % b + b) % b; // % is remainder not modulus
}

void play (char *board, char *newboard, int N, int M, int MPIRank, int MPISize, char *topRecv, char *bottomRecv) {
  int i, j, a;
  struct timeval startwtime, endwtime;

  //======================MPI =======================
  int topRank    = modulus(MPIRank - 1, MPISize);
  int bottomRank = modulus(MPIRank + 1, MPISize);

  MPI_Status reqStatuses[4];
  MPI_Request req[4];

  MPI_Isend(board,          M, MPI_CHAR, topRank,    0, MPI_COMM_WORLD, &req[0]); // first row
  MPI_Isend(&Board(N-1, 0), M, MPI_CHAR, bottomRank, 1, MPI_COMM_WORLD, &req[1]); // last  row
  MPI_Irecv(topRecv,        M, MPI_CHAR, topRank,    1, MPI_COMM_WORLD, &req[2]); // tags not needed?
  MPI_Irecv(bottomRecv,     M, MPI_CHAR, bottomRank, 0, MPI_COMM_WORLD, &req[3]);
  //====================================================

  /* apply the rules of Life to INTERNAL cells*/
  gettimeofday(&startwtime, NULL);
  #pragma omp parallel for private(i, j, a) collapse(2)
  for (i=1; i<N-1; i++) { //skip first and last
    for (j=0; j<M; j++) {
      a = adjacent_to(board, i, j, N, M);
      NewBoard(i, j) = calculateCell(Board(i,j), a);
    }
  }
  gettimeofday(&endwtime, NULL);
  printf("finished inside %d: %fs\n", MPIRank, (double)((endwtime.tv_usec - startwtime.tv_usec)
        /1.0e6 + endwtime.tv_sec - startwtime.tv_sec));

  /* apply the rules of Life to EXTERNAL cells*/
  gettimeofday(&startwtime, NULL);
  MPI_Waitall(2, &req[2], &reqStatuses[2]);
  calculateRow(board, newboard, N , M, 0,   topRecv);
  calculateRow(board, newboard, N , M, N-1, bottomRecv);
  gettimeofday(&endwtime, NULL);
  printf("waiting edges time rank %d: %fs\n", MPIRank, (double)((endwtime.tv_usec - startwtime.tv_usec)
        /1.0e6 + endwtime.tv_sec - startwtime.tv_sec));

  /* COPY the new board back into the old board */
  gettimeofday(&startwtime, NULL);
  #pragma omp parallel for private(i, j) collapse(2)
  for (i=0; i<N; i++) {
    for (j=0; j<M; j++) {
      Board(i,j) = NewBoard(i,j);
    }
  }
  gettimeofday(&endwtime, NULL);
  printf("finished copying %d: %fs\n", MPIRank, (double)((endwtime.tv_usec - startwtime.tv_usec)
        /1.0e6 + endwtime.tv_sec - startwtime.tv_sec));

  MPI_Waitall(2, req, reqStatuses);
}


