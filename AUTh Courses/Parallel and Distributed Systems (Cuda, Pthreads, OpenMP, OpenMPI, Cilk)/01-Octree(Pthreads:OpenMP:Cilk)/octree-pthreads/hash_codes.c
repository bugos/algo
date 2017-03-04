#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "float.h"
#include "pthread.h"

#define DIM 3

inline unsigned int compute_code(float x, float low, float step){

  return floor((x - low) / step);

}

struct quantize_data {
    unsigned int *codes;
    float *X;
    float *low;
    float step;
    int N;
};

/* Thread routine*/
void *pthread_quantize(void* pthread_data) {
  struct quantize_data *d = (struct quantize_data *) pthread_data;

  for(int i=0; i<d->N; i++){
    for(int j=0; j<DIM; j++){
      d->codes[i*DIM + j] = compute_code(d->X[i*DIM + j], d->low[j], d->step); 
    }
  }
  pthread_exit(0);
}

/* Function that does the quantization */
void quantize(unsigned int *codes, float *X, float *low, float step, int N) {
    extern int nthreads;

    // Pthread initialize
    pthread_t threads[nthreads];
    void *status;

    struct quantize_data args[nthreads];
    int work_size = N / nthreads, offset = 0;
    for(int tc = 0; tc < nthreads; tc++) {
        // Assign work to each thread
        args[tc].codes = &codes[offset];
        args[tc].X     = &X[offset];
        args[tc].low   = low;
        args[tc].step  = step;
        args[tc].N     = work_size;
        if ( nthreads - 1 == tc ) { //last thread takes remaining work
          args[tc].N = N - offset / DIM ;
        }
        offset += work_size * DIM; //offset next thread to correct position

        // Create thread
        int rc = pthread_create(&threads[tc], NULL, pthread_quantize, (void *)&args[tc]);
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

float max_range(float *x){

  float max = -FLT_MAX;
  for(int i=0; i<DIM; i++){
    if(max<x[i]){
      max = x[i];
    }
  }

  return max;

}

void compute_hash_codes(unsigned int *codes, float *X, int N, 
			int nbins, float *min, 
			float *max){
  
  float range[DIM];
  float qstep;

  for(int i=0; i<DIM; i++){
    range[i] = fabs(max[i] - min[i]); // The range of the data
    range[i] += 0.01*range[i]; // Add somthing small to avoid having points exactly at the boundaries 
  }

  qstep = max_range(range) / nbins; // The quantization step 
  
  quantize(codes, X, min, qstep, N); // Function that does the quantization

}



