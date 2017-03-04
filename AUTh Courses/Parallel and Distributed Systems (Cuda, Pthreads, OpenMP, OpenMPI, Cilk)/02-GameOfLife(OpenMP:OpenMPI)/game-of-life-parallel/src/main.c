/*
 * Game of Life implementation based on
 * http://www.cs.utexas.edu/users/djimenez/utsa/cs1713-3/c/life.txt
 * 
 */

//#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>
#include "omp.h"

#include <game-of-life.h>


int main (int argc, char *argv[]) {  
  if (argc != 6) { // Check if the command line arguments are correct 
    printf("Usage: %s N M thres disp\n"
     "where\n"
     "  N     : vertical  size of table (N)\n" // N always bigger
     "  M     : horizonal size of table (M)\n"
     "  thres : propability of alive cell\n"
     "  t     : number of generations\n"
     "  disp  : {1: display output, 0: hide output}\n"
           , argv[0]);
    return (1);
  }
  // Input command line arguments
  int N = atoi(argv[1]);        // Array size N
  int M = atoi(argv[2]);        // Array size N
  double thres = atof(argv[3]); // Propability of life cell
  int t = atoi(argv[4]);        // Number of generations 
  int disp = atoi(argv[5]);     // Display output?
    
  //======================MPI =======================
  int ierr, MPIRank, MPISize, providedThread;
  // ierr = MPI_Init(&argc, &argv); // Gia na xekinisi to MPI einai aparateto se kathe MPI programa
  // if fail abort
  ierr = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThread);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIRank); // To procid tha exi ton rank = arithmo tou pirina pou briskomaste
  MPI_Comm_size(MPI_COMM_WORLD, &MPISize); // to numprocs tha pari to posa cores iparxoun genikos  
  
  omp_set_num_threads(8);
  omp_set_nested(0);
  //=================================================
  printf("Hello from rank: %i of %i numtasks \n", MPIRank, MPISize);

  N = N / MPISize; // split table horizonally

  printf("Size %dx%d with propability: %0.1f%%\n", N, M, thres*100); //fix NxN

  int rowSize   =     M * sizeof(char);
  int boardSize = N * M * sizeof(char);
  char *board      = (char *)malloc(boardSize);
  char *newboard   = (char *)malloc(boardSize);
  char *topRecv    = (char *)malloc(rowSize);
  char *bottomRecv = (char *)malloc(rowSize);
  printf("%u\n", (unsigned int)board);
  printf("%u\n", (unsigned int)topRecv);

  //Barrier needed before getting time for all processes.
  struct timeval startwtime, endwtime;
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&startwtime, NULL);

  initialize_board (board, N, M);printf("Board initialized\n");// 0 se ola ta stixia tou pinaka 
  generate_table (board, N, M, thres, MPIRank);printf("Board generated\n");
  
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&endwtime, NULL);
  double init_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
        /1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("Init execution time : %fs\n", init_time);  

  int i;
  for (i=0; i<t; i++) { 
    printf("Game running %d generation.\n", i);
    if (disp) display_table (board, N,M);
    play (board, newboard, N, M, MPIRank, MPISize, topRecv, bottomRecv);  
  }
  printf("Game finished after %d generations.\n\n", t);

  //Barrier needed before getting time for all processes.
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&endwtime, NULL);
  double total_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
        /1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("Total execution time for all processes : %fs\n", total_time);

  // Destructor
  free(board);
  free(newboard);
  MPI_Finalize();
  return 0;
}
