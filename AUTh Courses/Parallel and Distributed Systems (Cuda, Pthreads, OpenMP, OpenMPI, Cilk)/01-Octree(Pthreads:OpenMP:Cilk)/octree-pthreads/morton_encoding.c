#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "pthread.h"

#define DIM 3

inline unsigned long int splitBy3(unsigned int a){
    unsigned long int x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

inline unsigned long int mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z){
    unsigned long int answer;
    answer = splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
}


struct morton_data {
    unsigned long int *mcodes;
    unsigned int *codes;
    int N;
    int max_level;
};

/* Thread routine*/
void *pthread_morton(void* pthread_data) {
  struct morton_data *d = (struct morton_data *) pthread_data;

  for(int i = 0; i < d->N; i++) {
    d->mcodes[i] = mortonEncode_magicbits(d->codes[i*DIM], d->codes[i*DIM + 1], d->codes[i*DIM + 2]);
  }

  pthread_exit(0);
}

/* The function that transform the morton codes into hash codes */ 
void morton_encoding(unsigned long int *mcodes, unsigned int *codes, int N, int max_level){
  extern int nthreads;

    // Pthread initialize
  pthread_t threads[nthreads];
  struct morton_data args[nthreads];
  int work_size = N / nthreads, offset = 0, offset2 = 0;
  for(int tc = 0; tc < nthreads; tc++) {
    // Assign work to each thread
    args[tc].mcodes    = &mcodes[offset];
    args[tc].codes     = &codes[offset2];
    args[tc].N         = work_size;
    args[tc].max_level = max_level;
    if ( nthreads - 1 == tc ) { //last thread takes remaining work
      args[tc].N = N - offset;
    }
    offset += work_size; //offset next thread to correct position
    offset2 += work_size * DIM; //offset for DIM*N sized codes[]

    // Create thread
    int rc = pthread_create(&threads[tc], NULL, pthread_morton, (void *)&args[tc]);
    if ( rc ) {
        printf("Error: pthread_create with code %d\n", rc);
        return;
    }
  }

  // Join Threads
  for (int tc = 0; tc < nthreads; tc++) {
    int rc = pthread_join(threads[tc], NULL);
    if(rc) {
      printf("Error: pthread_join with code %d\n", rc);
      return;
    }
  }
}
  
