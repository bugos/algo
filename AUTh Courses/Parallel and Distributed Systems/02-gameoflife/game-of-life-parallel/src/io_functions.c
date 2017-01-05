#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <game-of-life.h>

/* print the life board */

void print (char *board, int N, int M) {
  int   i, j;
  /* for each row */
  for (i=0; i<N; i++) {
    /* print each column position... */
      for (j=0; j<M; j++) {
      printf ("%c", Board(i,j) ? 'x' : ' ');
    }

    /* followed by a carriage return */
    printf ("\n");
  }
}



/* display the table with delay and clear console */

void display_table(char *board, int N, int M) {
  print (board, N, M);
  usleep(100000);  
  /* clear the screen using VT100 escape codes */
  puts ("\033[H\033[J");
}
