#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_
#include "vector_dot_product.h"

__device__ void lock(int *mutex)
{
	while(atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex)
{
	atomicExch(mutex, 0);
}

__global__ void vector_dot_product_kernel(float *A_on_host, float *B_on_host, float* result, int numElements, int *mutex)
{
	__shared__ float tempRes[TILE_SIZE];
	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	int strideLen = blockDim.x * gridDim.x;
	float localSum = 0.0f;	
	
	while( i < numElements) 
	{
        localSum += A_on_host[i] * B_on_host[i];
        i += strideLen;
    }
	__syncthreads();
    tempRes[threadIdx.x] = localSum;
    __syncthreads();
	
	i = TILE_SIZE / 2;
	while ( i != 0 )
	{
		if ( threadIdx.x < i ) 
			tempRes[threadIdx.x] += tempRes[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}
	
    if ( threadIdx.x == 0 ) 
	{
        lock(mutex);
        result[0] += tempRes[0];
        unlock(mutex);
    }
}

#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
