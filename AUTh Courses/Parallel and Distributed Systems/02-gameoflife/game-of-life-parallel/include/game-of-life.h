/* #ifndef UTILS_H_   /\* Include guard *\/ */
/* #define UTILS_H_ */

#define Board(x,y) board[(x)*N + (y)]
#define NewBoard(x,y) newboard[(x)*N + (y)]

/* set everthing to zero */

void initialize_board (char *board, int N, int M);

/* add to a width index, wrapping around like a cylinder */

int xadd (int i, int a, int N);

/* return the number of on cells adjacent to the i,j cell */
int adjacent_to (char *board, int i, int j, int N, int M);
int adjacent_to_edge (char *board, int i, int j, int N, int M, char *edgeRow);

/* play the game through one generation */

void play (char *board, char *newboard, int N, int M, int MPIRank, int MPISize,  char *topRecv, char *bottomRecv);

/* print the life board */

void print (char *board, int N, int M);

/* generate random table */

void generate_table (char *board, int N, int M, float threshold, int rank);

/* display the table with delay and clear console */

void display_table(char *board, int N, int M);

/* #endif // FOO_H_ */
