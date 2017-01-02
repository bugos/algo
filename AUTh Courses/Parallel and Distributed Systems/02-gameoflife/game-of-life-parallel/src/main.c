/*
 * Game of Life implementation based on
 * http://www.cs.utexas.edu/users/djimenez/utsa/cs1713-3/c/life.txt
 * 
 */

//#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>

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
  int N = atoi(argv[2]);        // Array size N
  double thres = atof(argv[3]); // Propability of life cell
  int t = atoi(argv[4]);        // Number of generations 
  int disp = atoi(argv[5]);     // Display output?
  
  struct timeval startwtime, endwtime;
  
  //======================MPI =======================
  int ierr, MPIRank, MPISize;
  ierr = MPI_Init(&argc, &argv); // Gia na xekinisi to MPI einai aparateto se kathe MPI programa
  // if fail abort

  //set a barrier to start counting at the same condition for every process
  MPI_Barrier(MPI_COMM_WORLD);
  //start time!
  gettimeofday (&startwtime, NULL);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &MPIRank); // To procid tha exi ton rank = arithmo tou pirina pou briskomaste
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &MPISize); // to numprocs tha pari to posa cores iparxoun genikos  
  
  printf("Hello from rank: %i of %i numtasks \n", MPIRank, MPISize);
  //=================================================

  int N = N / MPISize; // split table horizonally

  printf("Size %dx%d with propability: %0.1f%%\n", N, N, thres*100); //fix NxN

  int rowSize = M * sizeof(int);
  int boardSize = N * M * sizeof(int);
  int *board    = (int *)malloc(boardSize);
  int *newboard = (int *)malloc(boardSize);

  initialize_board (board, N,M);printf("Board initialized\n");// 0 se ola ta stixia tou pinaka 
  generate_table (board, N,M, thres, Rank);printf("Board generated\n");

  for (i=0; i<t; i++) { 
    if (disp) display_table (board, N,M);
    play (board, newboard, N, M, MPIRank, MPISize);  
  }

  printf("Game finished after %d generations.\n\n", t);
  //set a barrier to stop timer after all processes have reached the same point
  MPI_Barrier(MPI_COMM_WORLD);
  //stop timer!
  gettimeofday (&endwtime, NULL);

  free(board);
  free(newboard);
  MPI_Finalize();

  printf("Total execution time for all processes : %fs\n", total_time);
  return 0;
}
