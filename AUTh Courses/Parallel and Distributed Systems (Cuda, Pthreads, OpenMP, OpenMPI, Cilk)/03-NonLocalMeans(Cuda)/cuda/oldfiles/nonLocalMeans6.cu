#include <math.h>
// #include <stdio.h>
#include "cuda_runtime.h"
#define NDEBUG1
#include <assert.h>

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y

#define NN 64
#define multi 1
#define BSZ0 16
#define BSZ (multi * (BSZ0))
#define GSZ (NN / BSZ)

#define MAX_NSZ 7
#define NSZ 5
#define NSZ2 (NSZ / 2)

#define BSZ_HL (BSZ + NSZ - 1)

#define SER(row, col, ncols) ((row) * (ncols) + (col))

__constant__ float gaussDistW[MAX_NSZ * MAX_NSZ];

// me 4*4 blocks kai kai 1024 thr/block kai gia 4 pixel/thread vgainei ligo panw apo 32k shared
// alla den mporw na exw 1024 thread, mallon giati einai mexri 768.
// ara to spaw se 256/block? -> ipologismos metaforwn.

__device__ __forceinline__ void getSharedBlock(float sharedImg[BSZ_HL][BSZ_HL], const float *globalImg, int I0, int N) {
	int ii = SER(ty, tx, BSZ0); // 2d to 1d index of thread i in the block
	do {
		int I  = ii % BSZ_HL; // x index in block including padding
		int J  = ii / BSZ_HL; // y index in block including padding
		int IGlobal = I0 + SER(J, I, N);      // global input index
		assert(I < BSZ_HL);
		if( (I < BSZ_HL) && (J < BSZ_HL) && (ii < N * N) ) {
			sharedImg[I][J] = globalImg[IGlobal]; // download from global
		}
		ii += BSZ0 * BSZ0; // next iteration starts THREADNUM position after
	} while ( ii < BSZ_HL * BSZ_HL ); // only J check needed ? 
}
__device__ __forceinline__ void getWeight(float blockImg[BSZ_HL][BSZ_HL], float foreignBlockImg[BSZ_HL][BSZ_HL], float sigma, float weightSum[multi][multi], float fSum[multi][multi]) {
	// Compute block weights with self
	// new tx is (1) blockdim away
	#define txM (tx + ( mx ) * blockDim.x)
	#define tyM (ty + ( my ) * blockDim.y)
	for (int mx = 0; mx < multi; mx++) { // Multiple pixels per thread
		for (int my = 0; my < multi; my++) {
			for(int k = 0; k < BSZ; k++ ) { // Other block
				for(int l = 0; l < BSZ; l++ ) {
					float partialW = 0;
					for(int m = -NSZ2; m <= NSZ2; m++) // Neighbourhoud
						for(int n = -NSZ2; n <= NSZ2; n++)
							partialW += gaussDistW[ SER((n + MAX_NSZ / 2), (m + MAX_NSZ / 2), MAX_NSZ)]
								* powf( ( 0.7//blockImg[(txM + NSZ2) + m][(tyM + NSZ2) + n] 
								 - foreignBlockImg[(k   + NSZ2) + m][(l   + NSZ2) + n] ), 2);
					// if (!mx && !my && k==1) printf("%f\n",partialW);
					partialW = expf((-partialW / sigma));
					weightSum[mx][my] += partialW;
					fSum[mx][my]      += partialW * foreignBlockImg[k + NSZ2][l + NSZ2];
				}
			}
		}
	}
}

__device__ __forceinline__ void downloadAndCalculate(float blockImg[BSZ_HL][BSZ_HL], float foreignBlockImg[BSZ_HL][BSZ_HL],
	const float *inputImg, float sigma, float weightSum[multi][multi], float fSum[multi][multi], int N, int I0) {
		getSharedBlock(foreignBlockImg, inputImg, I0, N);
		__syncthreads();
		getWeight(blockImg, foreignBlockImg, sigma, weightSum, fSum);
}


__global__ void nlm(float const *inputImg, float *outputImg, int N, float sigma) {
	// assert(NN == N);
	int N2 = NN + NSZ - 1; // input image with padding

	// assert(GSZ == gridDim.x);
	// assert(BSZ0 == blockDim.x);
	__shared__ float        blockImg[BSZ_HL][BSZ_HL];
	__shared__ float foreignBlockImg[BSZ_HL][BSZ_HL];
	
	// if(!tx && !ty && !bx && !by) {
	// for (int mu = 0; mu < 49; mu++) {
	// 	printf("%f ",gaussDistW[mu]);
	// }
	// }
	// __syncthreads();

	float weightSum[multi][multi], fSum[multi][multi]; // Weightsums for multiple pixels per thread.
	for (int mx = 0; mx < multi; mx++) { 
		for (int my = 0; my < multi; my++) {
			weightSum[mx][my] = 0;
			fSum[mx][my] = 0;
		}
	}

	// put inside
	int I0 = SER(by * BSZ, bx * BSZ, N2); // Download this block's pixels
	//downloadAndCalculate(blockImg, blockImg, inputImg, sigma, weightSum, fSum, N2, I0);
	for (char i = 0; i < GSZ; i++) { // gia kathe BLOCK stin arxiki eikona (X)
		for (char j = 0; j < GSZ; j++) {
			if ( !(by == j && bx == i) ) {
				// Download other blocks
				int I1 = SER(j * BSZ, i * BSZ, N2); //first pixel in block. Used as a ref point to calculate the block.(pg21)
				downloadAndCalculate(blockImg, foreignBlockImg, inputImg, sigma, weightSum, fSum, N2, I1);
			}
		}
	}

	for (int mx = 0; mx < multi; mx++) { 
		for (int my = 0; my < multi; my++) {
			// add NSZ2 to skip the padding pixels
			outputImg[SER(by * BSZ, bx * BSZ, NN) + SER(tyM, txM, NN)] = fSum[mx][my] / weightSum[mx][my];
			//blockImg[(txM + NSZ2)][(tyM + NSZ2)]
			//inputImg[I0 + SER(NSZ2, NSZ2, N2) + SER(tyM, txM, NN)]
		}
	}
}

// template __global__ void kernel<false>();