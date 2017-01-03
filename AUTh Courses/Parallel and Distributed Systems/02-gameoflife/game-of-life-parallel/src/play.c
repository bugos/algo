#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include "omp.h"

#include <game-of-life.h>


void inline calculateRow(int *board, int *newboard, int N , int M, int i, int *edgeRow) {
  //#pragma omp parallel for
  for (j=0; j<M; j++) {
    int a = adjacent_to(board, i, j, N, M, bottom_recv);
    calculateCell(board, newboard, N , M, 0  ,    );
  }
}

void inline calculateCell(int *cell, int *newcell, int a) {
    if (a == 2) NewBoard(i,j) = Board(i,j);
    if (a == 3) NewBoard(i,j) = 1;
    if (a < 2)  NewBoard(i,j) = 0; // else
    if (a > 3)  NewBoard(i,j) = 0; 

}

void play (int *board, int *newboard, int N, int M, int MPIRank, int MPISize) {
  /*
    1.STASIS : If, for a given cell, the number of on neighbours is 
    exactly two, the cell maintains its status quo into the next 
    generation. If the cell is on, it stays on, if it is off, it stays off.

    2.GROWTH : If the number of on neighbours is exactly three, the cell 
    will be on in the next generation. This is regardless of the cell's
    current state.

    3.DEATH : If the number of on neighbours is 0, 1, 4-8, the cell will 
    be off in the next generation.
  */
  int i, j;

  //omp_set_num_threads(8);

  //======================MPI =======================
  int *top_send    = (int *)malloc(rowSize); //move to main
  int *bottom_send = (int *)malloc(rowSize);
  int *top_recv    = (int *)malloc(rowSize);
  int *bottom_recv = (int *)malloc(rowSize);
  
  memcpy(top_send,    board,                       rowSize);
  memcpy(bottom_send, board + boardSize - rowSize, rowSize);
  
  int topRank    = (MPIrank - 1) % MPISize;
  int bottomRank = (MPIrank + 1) % MPISize;

  MPI_Status mpistat;
  MPI_Request top_recv_request, bottom_recv_request, top_send_request, bottom_send_request;
  MPI_Isend(top_send,    M, MPI_INT, topRank,    0, MPI_COMM_WORLD, &top_send_request   );
  MPI_Isend(bottom_send, M, MPI_INT, bottomRank, 1, MPI_COMM_WORLD, &bottom_recv_request);
  MPI_Irecv(top_recv,    M, MPI_INT, topRank,    0, MPI_COMM_WORLD, &top_recv_request   );
  MPI_Irecv(bottom_recv, M, MPI_INT, bottomRank, 1, MPI_COMM_WORLD, &bottom_send_request);
  //====================================================

  /* apply the rules of Life to INTERNAL cells*/
  //omp_set_nested(false)
  //#pragma omp parallel for
  for (i=1; i<N-1; i++) { //skip first and last
    for (j=0; j<M; j++) { 
      int a = adjacent_to(board, i, j, N, M, );
      calculateCell(board, newboard, N , M, i);
    }
  }

  /* apply the rules of Life to EXTERNAL cells*/
  MPI_Wait(&top_recv_request,    &mpistat);
  MPI_Wait(&bottom_recv_request, &mpistat);
  calculateRow(board, newboard, N , M, 0,   top_recv);
  calculateRow(board, newboard, N , M, N-1, bottom_recv);
  
  /*Swap the board in main after the play  
    int *temp = newboard;
    newboard = board;
    board = temp;*/
  
  /* COPY the new board back into the old board */
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      Board(i,j) = NewBoard(i,j);
    }
  }

  MPI_Wait(&top_send_request,    &mpistat);
  MPI_Wait(&bottom_send_request, &mpistat);
}


