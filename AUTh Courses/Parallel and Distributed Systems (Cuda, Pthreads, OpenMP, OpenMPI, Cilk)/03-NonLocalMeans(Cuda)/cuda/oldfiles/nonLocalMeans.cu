#include <math.h>
#include <stdio.h>

#define BSZ = blockDim.x //This variable and contains the dimensions of the block.
#define GSZ = gridDim.x  //This variable contains the dimensions of the grid.
#define tx  = threadIdx.x//This variable contains the thread index within the block
#define ty  = threadIdx.y
#define bx  = blockIdx.x //This variable contains the block index within the grid.
#define by  = blockIdx.y

__constant__ int gaussDistW[7 * 7];

__global__ void nlm(float const *inputImg, float *outputImg, int N, int neighSize, float sigma) {
	__shared__ float        blockImg[BSZ+neighSize][BSZ+neighSize];
	__shared__ float foreignBlockImg[BSZ+neighSize][BSZ+neighSize];

	int weightSum = 0, fSum = 0;

	int I0 = by * BSZ * N + bx * BSZ;
	getSharedBlock(blockImg, inputImg, I0, neighSize, N);
	__syncthreads();
	getWeight(blockImg, blockImg, neighSize, sigma, &weightSum, &fSum);
	
	// Compute block weights with other blocks
	for (int i = 0; i < GSZ; i++) {
		for (int j = 0; j < GSZ; j++) { // gia kathe BLOCK stin arxiki eikona (X)
			int I1 = j * BSZ * N + i * BSZ; // jN+i, find block
			if ( I1 == I0 ) continue;
			getSharedBlock(foreignBlockImg, inputImg, I1, neighSize, N);
			__syncthreads();
			getWeight(blockImg, foreignBlockImg, neighSize, sigma, &weightSum, &fSum);
		}
	}
	outputImg[I0 + (ty + neighSize2) * N + (tx + neighSize2)] = fSum / weightSum;
}

// inline
__device__ inline void getSharedBlock(float **sharedImg, const float *globalImg, int I0, int neighSize, int N) {
	#define BSZ_HALO (BSZ + neighSize - 1)
	// Load the first thrednum values
	int ii = ty * BSZ + tx;              // 2d to 1d index of thread
	int I  = ii % BSZ_HALO; // x index including padding
	int J  = ii / BSZ_HALO; // y index including padding
	int IGlobal = I0 + J * N + I;      // global input index
	sharedImg[I][J] = globalImg[IGlobal];

	// Load the remaining values
	int ii2 = BSZ * BSZ + ty * BSZ + tx; // 2d to 1d thread starting
	int I2  = ii2 % BSZ_HALO; // x index including padding
	int J2  = ii2 / BSZ_HALO; // y index including padding
	 int  IGlobal2 =  I0  +  J2 * N  +  I2;  // N+padding???
	if ( ( I2 < BSZ_HALO )  &&  ( J2 < ( BSZ_HALO )  &&  ( ii2  <  N * N ) )
		sharedImg[I2][J2] = globalImg[IGlobal2];
	__syncthreads();
}


__device__ inline void getWeight(float **blockImg, float **foreignBlockImg, int neighSize, float sigma, float *weightSum, float *fSum) {
	// Compute block weights with self
	for(int k = 0; k < BSZ; k++ ) {
		for(int l = 0; l < BSZ; l++ ) {
			int weight = 0;
			#define neighSize2 (neighSize / 2);
			for(int m = -neighSize2; m <= neighSize2; m++)
				for(int n = -neighSize2; n <= neighSize2; n++)
					weight += gaussDistW[ (n + neighSize2) * neighSize + (m + neighSize2)] 
							* pow( blockImg[(tx + neighSize2) + m][(ty + neighSize2) + n] 
						  - foreignBlockImg[(k  + neighSize2) + m][(l  + neighSize2) + n], 2);
			weight = exp(-weight / sigma);
			weightSum += weight;
			fSum      += weight * foreignBlockImg[k][l]
		}
	}
}

	// // Load the first thrednum values
	// int ii = j * BSZ + i;              // 2d to 1d index of thread
	// int I  = ii % ( BSZ + neighSize - 1 ); // x index including padding
	// int J  = ii / ( BSZ + neighSize - 1); // y index including padding
	// int IGlobal = I0 + J * N + I;      // global input index
	// blockImg[I][J] = inputImg[IGlobal];

	// // Load the remaining values
	// int ii2 = BSZ * BSZ + j * BSZ + i; // 2d to 1d thread starting
	// int I2  = ii2 % ( BSZ + neighSize - 1 ); // x index including padding
	// int J2  = ii2 / ( BSZ + neighSize - 1 ); // y index including padding
	//  int  IGlobal2 =  I0  +  J2 * N  +  I2;  // N+padding???
	// if ( ( I2 < ( BSZ + 2 ) )  &&  ( J2 < ( BSZ + 2 ) )  &&  ( ii2  <  N * N ) )
	// 	blockImg[I2][J2] = inputImg[IGlobal2];