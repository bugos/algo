#include <math.h>
#include <stdio.h>

#define BSZ (2 * blockDim.x)
#define BSZ0 blockDim.x
#define GSZ gridDim.x
#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y

#define MAX_NSZ 7
#define neighSize2 (neighSize / 2)
#define BSZ_HL0 (BSZ + neighSize - 1)
#define BSZ_HL (2 * (64) + (7) - 1)

extern __constant__ int gaussDistW[MAX_NSZ * MAX_NSZ];

// inline
__device__ inline void getSharedBlock(float sharedImg[BSZ_HL][BSZ_HL], const float *globalImg, int I0, int neighSize, int N) {
	// Load the first thrednum values
	int ii = ty * BSZ0 + tx;              // 2d to 1d index of thread
	int I  = ii % BSZ_HL; // x index including padding
	int J  = ii / BSZ_HL; // y index including padding
	int IGlobal = I0 + J * N + I;      // global input index
	sharedImg[I][J] = globalImg[IGlobal];

	// Load the remaining values
	int ii2 = BSZ0 * BSZ0 + ty * BSZ0 + tx; // 2d to 1d thread starting
	int I2  = ii2 % BSZ_HL; // x index including padding
	int J2  = ii2 / BSZ_HL; // y index including padding
	 int  IGlobal2 =  I0  +  J2 * N  +  I2;  // N+padding???
	if ( ( I2 < BSZ_HL )  &&  ( J2 < ( BSZ_HL )  ) &&  ( ii2  <  N * N ) )
		sharedImg[I2][J2] = globalImg[IGlobal2];
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

	int weightSum = 0, fSum = 0;

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
