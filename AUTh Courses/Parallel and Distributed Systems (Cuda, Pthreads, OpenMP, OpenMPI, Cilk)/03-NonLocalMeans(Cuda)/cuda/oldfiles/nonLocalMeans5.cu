#include <math.h>
#include <stdio.h>
#define NDEBUG
#include <assert.h>

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y

//           multi/2 * blockDim.x
#define NN 64
#define multi 2
#define BSZ0 16
#define BSZ (multi * (BSZ0))
#define GSZ (NN / BSZ)


#define MAX_NSZ 7
#define NSZ 5
#define NSZ2 (NSZ / 2)

#define BSZ_HL (BSZ + NSZ - 1)
//->use constant values in neighsize and bsz independently

#define SER(row, col, ncols) ((row) * (ncols) + (col))

extern __constant__ int gaussDistW[MAX_NSZ * MAX_NSZ];

// me 4*4 blocks kai kai 1024 thr/block kai gia 4 pixel/thread vgainei ligo panw apo 32k shared
// alla den mporw na exw 1024 thread, mallon giati einai mexri 768.
// ara to spaw se 256/block? -> ipologismos metaforwn.

__device__ __forceinline__ void getSharedBlock(float sharedImg[BSZ_HL][BSZ_HL], const float *globalImg, int I0, int N) {
	int ii = SER(ty, tx, BSZ0); // 2d to 1d index of thread i in the block
	do {
		int I  = ii % BSZ_HL; // x index in block including padding
		int J  = ii / BSZ_HL; // y index in block including padding
		int IGlobal = I0 + SER(J, I, N);      // global input index
		if( (I < BSZ_HL) && (J < BSZ_HL) && (ii < N * N) ) {
			sharedImg[I][J] = globalImg[IGlobal]; // download from global
		}
		ii += BSZ0 * BSZ0; // next iteration starts THREADNUM position after
	} while ( ii < N * N ); // only J check needed ? 
}
__device__ __forceinline__ void getWeight(float blockImg[BSZ_HL][BSZ_HL], float foreignBlockImg[BSZ_HL][BSZ_HL], float sigma, float weightSum[multi * multi], float fSum[multi * multi]) {
	// Compute block weights with self
	// float partialW;
	// new tx is (1) blockdim away
	#define txM (tx + ( mu % multi ) * blockDim.x) 
	#define tyM (ty + ( mu / multi ) * blockDim.y)
	for (int mu = 0; mu < multi * multi; mu++) { // Multiple pixels per thread
		for(char k = 0; k < BSZ; k++ ) { // Other block
			for(char l = 0; l < BSZ; l++ ) {
				float partialW = 0;
				for(char m = -NSZ2; m <= NSZ2; m++) // Neighbourhoud
					for(char n = -NSZ2; n <= NSZ2; n++)
						partialW += gaussDistW[ (n + MAX_NSZ / 2) * MAX_NSZ + (m + MAX_NSZ / 2)]
							* powf( ( blockImg[(txM + NSZ2) + m][(tyM + NSZ2) + n] 
							 - foreignBlockImg[(k   + NSZ2) + m][(l   + NSZ2) + n] ), 2);
				partialW = expf((-partialW / sigma));
				weightSum[mu] += partialW;
				fSum[mu]      += partialW * foreignBlockImg[k][l];
			}
		}
	}
}

__device__ __forceinline__ void downloadAndCalculate(float blockImg[BSZ_HL][BSZ_HL], float foreignBlockImg[BSZ_HL][BSZ_HL],
	const float *inputImg, float sigma, float *weightSum, float *fSum, int N, int I0) {
		getSharedBlock(foreignBlockImg, inputImg, I0, N);
		__syncthreads();
		getWeight(blockImg, foreignBlockImg, sigma, weightSum, fSum);
}


__global__ void nlm(float const *inputImg, float *outputImg, int N, float sigma) {
	N = N + NSZ - 1; // image with padding
	// assert(NN == N);
	// assert(GSZ == gridDim.x);
	// assert(BSZ0 == blockDim.x);
	// printf("HELLO CUDA");

	__shared__ float        blockImg[BSZ_HL][BSZ_HL];
	__shared__ float foreignBlockImg[BSZ_HL][BSZ_HL];
	
	float weightSum[multi * multi], fSum[multi * multi]; // Weightsums for multiple pixels per thread.
	for (int mu = 0; mu < multi * multi; mu++) {
		weightSum[mu] = 0;
		fSum[mu] = 0;
	}

	int I0 = SER(by * BSZ, bx * BSZ, N); // Download this block's pixels
	downloadAndCalculate(blockImg, blockImg, inputImg, sigma, weightSum, fSum, N, I0);
	for (char i = 0; i < GSZ; i++) { // gia kathe BLOCK stin arxiki eikona (X)
		for (char j = 0; j < GSZ; j++) {
			if ( !(by == j && bx == i) ) {
				// Download other blocks
				int I1 = SER(j * BSZ, i * BSZ, N); //first pixel in block. Used as a ref point to calculate the block.(pg21)
				downloadAndCalculate(blockImg, foreignBlockImg, inputImg, sigma, weightSum, fSum, N, I1);
			}
		}
	}

	for (int mu = 0; mu < multi * multi; mu++) { // Multiple pixels per thread
		// add NSZ2 to skip the padding pixels
		outputImg[I0 + (tyM + NSZ2) * N + (txM + NSZ2)] = fSum[mu] / weightSum[mu];
	}
}

// template __global__ void kernel<false>();