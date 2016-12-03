#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pthread.h"


#define DIM 3

struct rearrangement_data {
    float *Y;
    float *X;
    unsigned int *permutation_vector;
    int N;
};

/* Thread routine*/
void *pthread_rearrangement(void* pthread_data) {
  struct rearrangement_data *d = (struct rearrangement_data *) pthread_data;

  for(int i = 0; i < d->N; i++) {
    memcpy(&d->Y[i*DIM], &d->X[d->permutation_vector[i]*DIM], DIM*sizeof(float));
  }

  pthread_exit(0);
}

void data_rearrangement(float *Y, float *X, unsigned int *permutation_vector, int N) {
	extern int nthreads;

    // Pthread initialize
  pthread_t threads[nthreads];
  struct rearrangement_data args[nthreads];
  int work_size = N / nthreads, offset = 0, offset2 = 0;

  for(int tc = 0; tc < nthreads; tc++){
    // Assign work to each thread
    args[tc].Y                  = &Y[offset2];
    args[tc].X                  = X;
    args[tc].permutation_vector = &permutation_vector[offset];
    args[tc].N                  = work_size;
    if ( nthreads - 1 == tc ) { //last thread takes remaining work
      args[tc].N = N - offset;
    }
    offset += work_size; //offset next thread to correct position
    offset2 += work_size * DIM; //offset for DIM*N sized codes[]

    // Create thread
    int rc = pthread_create(&threads[tc], NULL, pthread_rearrangement, (void *)&args[tc]);
    if ( rc ) {
        printf("Error: pthread_create with code %d\n", rc);
        return;
    }
  }

  // Join Threads
  for (int tc = 0; tc < nthreads; tc++) {
      int rc = pthread_join(threads[tc], NULL);
      if (rc) {
          printf("Error: pthread_join with code %d\n", rc);
          return;
      }
  }
}
