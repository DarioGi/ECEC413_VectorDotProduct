#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include "vector_dot_product.h"

// includes, kernels
#include "vector_dot_product_kernel.cu"

void run_test(unsigned int);
float compute_on_device(float *, float *,int);
void check_for_error(char *);
extern "C" float compute_gold( float *, float *, unsigned int);
float* allocate_vector_on_gpu(int numOfElements);
void copy_vector_to_device(float* hostVector, float* gpuVector, int numOfElements);
void copy_vector_from_device(float* hostVector, float* gpuVector, int numOfElements);

int 
main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: vector_dot_product <num elements> \n");
		exit(0);	
	}
	unsigned int num_elements = atoi(argv[1]);
	run_test(num_elements);
	return 0;
}

void 
run_test(unsigned int num_elements) 
{
	// Obtain the vector length
	unsigned int size = sizeof(float) * num_elements;

	// Allocate memory on the CPU for the input vectors A and B
	float *A = (float *)malloc(size);
	float *B = (float *)malloc(size);
	
	// Randomly generate input data. Initialize the input data to be floating point values between [-.5 , 5]
	printf("Generating random vectors with values between [-.5, .5]. \n");	
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; i++){
		A[i] = (float)rand()/(float)RAND_MAX - 0.5;
		B[i] = (float)rand()/(float)RAND_MAX - 0.5;
	}
	
	printf("Generating dot product on the CPU. \n");
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
	float reference = compute_gold(A, B, num_elements);
	
    gettimeofday(&stop, NULL);
	printf("Gold Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Edit this function to compute the result vector on the GPU. 
       The result should be placed in the gpu_result variable. */
	float gpu_result = compute_on_device(A, B, num_elements);

	printf("Result on CPU: %f, result on GPU: %f. \n", reference, gpu_result);
    printf("Epsilon: %f. \n", fabsf(reference - gpu_result));

	// cleanup memory
	free(A);
	free(B);
	
	return;
}

/* Edit this function to compute the dot product on the device using atomic intrinsics. */
float compute_on_device(float *A_on_host, float *B_on_host, int num_elements)
{
	float* dA = allocate_vector_on_gpu(num_elements);
	copy_vector_to_device(dA, A_on_host, num_elements);
	float* dB = allocate_vector_on_gpu(num_elements);
	copy_vector_to_device(dB, B_on_host, num_elements);
	float* dC = allocate_vector_on_gpu(1);
	cudaMemset( dC, 0.0f, sizeof(float));
	
	if ( dA == NULL || dB == NULL || dC == NULL )
	{
		printf("Unable to allocate memory!\n");
		return 0;
	}
	
	int *mutex = NULL;
    cudaMalloc((void **)&mutex, sizeof(int));
    cudaMemset(mutex, 0, sizeof(int));
	
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
	vector_dot_product_kernel<<<GRID_SIZE, TILE_SIZE>>>(dA, dB, dC, num_elements, mutex);
	
	cudaThreadSynchronize();
	
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err ) 
	{
		fprintf(stderr, "Kernel execution failed: %s.\n", cudaGetErrorString(err));
		return 0;
	}
	float res = 0.0f;
	copy_vector_from_device(&res, dC, 1);
	
	gettimeofday(&stop, NULL);
	
	printf("Shared Memory Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	cudaFree(mutex);
	
    return res;
}

float* allocate_vector_on_gpu(int numOfElements)
{
    float* vector = NULL;
    int size = numOfElements * sizeof(float);
    cudaMalloc((void**) &vector, size);
	if ( vector == NULL )
	{
		printf("Unable to allocate vector on GPU. Exiting...\n");
		exit(1);
	}
    return vector;
}


void copy_vector_to_device(float* gpuVector, float* hostVector, int numOfElements)
{
    int size = numOfElements * sizeof(float);
    cudaMemcpy(gpuVector, hostVector, size, cudaMemcpyHostToDevice);
}


void copy_vector_from_device(float* hostVector, float* gpuVector, int numOfElements)
{
    int size = numOfElements * sizeof(float);
    cudaMemcpy(hostVector, gpuVector, size, cudaMemcpyDeviceToHost);
}
 
// This function checks for errors returned by the CUDA run time
void 
check_for_error(char *msg)
{
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 
