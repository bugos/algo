#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "pthread.h"


#define MAXBINS 8

int open_threads;
pthread_mutex_t thread_mutex;


inline void swap_long(unsigned long int **x, unsigned long int **y){

  unsigned long int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;
}

inline void swap(unsigned int **x, unsigned int **y){

  unsigned int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

struct radix_data {
    unsigned long int *morton_codes;
    unsigned long int *sorted_morton_codes;
    unsigned int *permutation_vector;
    unsigned int *index;
    unsigned int *level_record;
    int N;
    int population_threshold;
    int sft;
    int lv;
};

// just the declaration
void truncated_radix_sort(
        unsigned long int *morton_codes, 
        unsigned long int *sorted_morton_codes, 
        unsigned int *permutation_vector,
        unsigned int *index,
        unsigned int *level_record,
        int N, 
        int population_threshold,
        int sft, 
        int lv);

void *pthread_radix_sort(void* pthread_data) {
  //retrieve pthread data
  struct radix_data *d = (struct radix_data *) pthread_data;
  truncated_radix_sort(
     d->morton_codes, 
     d->sorted_morton_codes, 
     d->permutation_vector, 
     d->index,
     d->level_record, 
     d->N, 
     d->population_threshold,
     d->sft,
     d->lv);

    // Thread finished
    pthread_mutex_lock(&thread_mutex);
    open_threads--;
    pthread_mutex_unlock(&thread_mutex);
    pthread_exit(0);
}

void truncated_radix_sort(
        unsigned long int *morton_codes, 
			  unsigned long int *sorted_morton_codes, 
			  unsigned int *permutation_vector,
			  unsigned int *index,
			  unsigned int *level_record,
			  int N, 
			  int population_threshold,
			  int sft, 
        int lv) {

  extern int nthreads; 

  int BinSizes[MAXBINS] = {0};
  int BinCursor[MAXBINS] = {0};
  unsigned int *tmp_ptr;
  unsigned long int *tmp_code;
  
  if (N<=0)
    return;

  if(N<=population_threshold || sft < 0) { // Base case. The node is a leaf

    level_record[0] = lv; // record the level of the node
    memcpy(permutation_vector, index, N*sizeof(unsigned int)); // Copy the pernutation vector
    memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int)); // Copy the Morton codes 

    return;
  }
  else{
    level_record[0] = lv;
    // Find which child each point belongs to 
    for(int j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      BinSizes[ii]++;
    }

    // scan prefix (must change this code)  
    int offset = 0;
    for(int i=0; i<MAXBINS; i++){
      int ss = BinSizes[i];
      BinCursor[i] = offset;
      offset += ss;
      BinSizes[i] = offset;
    }
    
    for(int j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      permutation_vector[BinCursor[ii]] = index[j];
      sorted_morton_codes[BinCursor[ii]] = morton_codes[j];
      BinCursor[ii]++;
    }
    
    //swap the index pointers  
    swap(&index, &permutation_vector);

    //swap the code pointers 
    swap_long(&morton_codes, &sorted_morton_codes);

    pthread_t threads[MAXBINS]; // array of threads
    struct radix_data args[MAXBINS]; //arguments array

    for(int i = 0; i < MAXBINS; i++){
        int offset = (i>0) ? BinSizes[i-1] : 0;
        int size = BinSizes[i] - offset;

        if (nthreads > open_threads && ) { //we have threads left

          // Assign work to each thread
          args[i].morton_codes = &morton_codes[offset];
          args[i].sorted_morton_codes = &sorted_morton_codes[offset];
          args[i].permutation_vector = &permutation_vector[offset];
          args[i].index = &index[offset];
          args[i].level_record = &level_record[offset];
          args[i].N = size;
          args[i].population_threshold = population_threshold;
          args[i].sft = sft-3;
          args[i].lv = lv+1;
          
          // Create thread
          int rc = pthread_create(&threads[i], NULL, pthread_radix_sort, (void *)&args[i]);
          if ( rc ) {
            printf("Error: pthread_create with code %d\n", rc);
            return;
          }
          else {
            pthread_mutex_lock(&thread_mutex);
            open_threads++;
            pthread_mutex_unlock(&thread_mutex);
          }
        }
        else { // no threads left
          threads[i] = (pthread_t)0; // no thread created
          truncated_radix_sort(
             &morton_codes[offset], 
             &sorted_morton_codes[offset], 
             &permutation_vector[offset], 
             &index[offset], 
             &level_record[offset], 
             size, 
             population_threshold,
             sft-3, 
             lv+1);
        }
    }

    // Join Threads
    for (int tc = 0; tc < MAXBINS; tc++) {
        if (!threads[tc]) {
          continue;
        }

        int rc = pthread_join(threads[tc], NULL);
        if (rc) {
            printf("Error: pthread_join with code %d\n", rc);
            return;
        }
    } 
  } 
}


