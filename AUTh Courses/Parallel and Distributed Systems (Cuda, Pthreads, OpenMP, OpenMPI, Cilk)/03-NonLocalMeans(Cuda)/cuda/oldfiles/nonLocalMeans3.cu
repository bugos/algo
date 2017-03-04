#include <math.h>
#include <stdio.h>

#define BSZ (1 * blockDim.x)
#define BSZ0 blockDim.x
#define GSZ gridDim.x
#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y

#define MAX_NSZ 7
#define neighSize2 (neighSize / 2)
#define BSZ_HL00 (BSZ + neighSize - 1)
#define BSZ_HL (1 * (64) + (7) - 1)
//->use constant values in neighsize and bsz independently

#define SER(row, col, ncols) ((row) * (ncols) + (col))

extern __constant__ int gaussDistW[MAX_NSZ * MAX_NSZ];

// me 4*4 blocks kai kai 1024 thr/block kai gia 4 pixel/thread vgainei ligo panw apo 32k shared
// alla den mporw na exw 1024 thread, mallon giati einai mexri 768.
// ara to spaw se 256/block? -> ipologismos metaforwn.

// inline
__device__ inline void getSharedBlock(float sharedImg[BSZ_HL][BSZ_HL], const float *globalImg, int I0, int neighSize, int N) {
	int ii = SER(ty, tx, BSZ0); // 2d to 1d index of thread
	do {
		int I  = ii % BSZ_HL; // x index including padding
		int J  = ii / BSZ_HL; // y index including padding
		int IGlobal = I0 + SER(J, I, N);      // global input index
		sharedImg[I][J] = globalImg[IGlobal];

		int ii += BSZ0 * BSZ0; // threadnum
	} while (( (I < BSZ_HL) && (J < BSZ_HL) && (ii < N * N) ));
}

__device__ inline void getSharedBlock0(float sharedImg[BSZ_HL][BSZ_HL], const float *globalImg, int I0, int neighSize, int N) {
	// Load the first thrednum values
	int ii = SER(ty, tx, BSZ0);              // 2d to 1d index of thread
	int I  = ii % BSZ_HL; // x index including padding
	int J  = ii / BSZ_HL; // y index including padding
	int IGlobal = I0 + SER(J, I, N);      // global input index
	sharedImg[I][J] = globalImg[IGlobal];

	// Load the remaining values
	int ii2 = BSZ0 * BSZ0 + SER(ty, tx, BSZ0); // 2d to 1d thread starting
	int I2  = ii2 % BSZ_HL; // x index including padding
	int J2  = ii2 / BSZ_HL; // y index including padding
	 int  IGlobal2 =  I0 +  SER(J2, I2, N + neighSize - 1);  // N+padding???
	if ( (I2 < BSZ_HL) && (J2 < BSZ_HL) && (ii2 < N * N) ) {
		sharedImg[I2][J2] = globalImg[IGlobal2];
	}
}


__device__ inline void getWeight(float blockImg[BSZ_HL][BSZ_HL], float foreignBlockImg[BSZ_HL][BSZ_HL], int neighSize, float sigma, float *weightSum, float *fSum) {
	// Compute block weights with self
	char k, l, m, n;
	for(k = 0; k < BSZ; k++ ) {
		for(l = 0; l < BSZ; l++ ) {
			int weight = 0;
			for(m = -neighSize2; m <= neighSize2; m++)
				for(n = -neighSize2; n <= neighSize2; n++)
					weight += gaussDistW[ (n + MAX_NSZ / 2) * MAX_NSZ + (m + MAX_NSZ / 2)] 
							* powf( ( blockImg[(tx + neighSize2) + m][(ty + neighSize2) + n] 
						  - foreignBlockImg[(k  + neighSize2) + m][(l  + neighSize2) + n] ), 2);
			weight = expf((-weight / sigma));
			*weightSum += weight;
			*fSum      += weight * foreignBlockImg[k][l];
		}
	}
}

__global__ void nlm(float const *inputImg, float *outputImg, int N, int neighSize, float sigma) {
	__shared__ float        blockImg[BSZ_HL][BSZ_HL];
	__shared__ float foreignBlockImg[BSZ_HL][BSZ_HL];

	float weightSum = 0, fSum = 0;

	N = N + neighSize - 1; // !!!
	int I0 = by * BSZ * N + bx * BSZ; // pg21
	getSharedBlock(blockImg, inputImg, I0, neighSize, N); // 4 times
	__syncthreads();
	getWeight(blockImg, blockImg, neighSize, sigma, &weightSum, &fSum); // 4 times
	
	char i, j;
	// Compute block weights with other blocks
	for (i = 0; i < GSZ / 2; i++) {
		for (j = 0; j < GSZ / 2; j++) { // gia kathe BLOCK stin arxiki eikona (X)
			int I1 = j * BSZ * N + i * BSZ; // jN+i, find block
			if ( I1 == I0 ) continue; // Don't recompute self
			getSharedBlock(foreignBlockImg, inputImg, I1, neighSize, N);
			__syncthreads();
			getWeight(blockImg, foreignBlockImg, neighSize, sigma, &weightSum, &fSum);
		}
	}
	outputImg[I0 + (ty + neighSize2) * N + (tx + neighSize2)] = fSum / weightSum;
}

// template __global__ void kernel<false>();