#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define imin(a,b) (a < b ? a : b)

namespace cuda_ros_node {

const int threadsPerBlock = 256;    // Arbitrary choice, max with 3070 GPU is 1024

__global__ void dot(float* a, float* b, float* c, int* arrSize)
{
    // Use device shared memory as a buffer to store each thread's running sum
    __shared__ float cache[threadsPerBlock];
    // Determine our indices
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int cacheIndex = threadIdx.x;
    // Perform the multiplication step of the dot product
    float temp = 0;
    while(tid < *arrSize)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    // Set the cache values
    cache[cacheIndex] = temp;
    // Synchronize all the threads running this in the whole block
    __syncthreads();
    // Perform the reduction step
    // Note: threadsPerBlock must be a power of 2 because of the following code
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if(cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        // Synchronize all threads running the reduction step
        __syncthreads();
        i /= 2;
    }
    // Final step: store the reduced summation
    if(cacheIndex == 0)
    {
        c[blockIdx.x] = cache[0];
    }
}

float cudaDot(float* a, float* b, int arrSize)
{
    // Calculate number of blocks we should launch
    // (smallest multiple of threadsPerBlock that is greater than or equal to arrSize)
    int blocksPerGrid = imin(32, (arrSize+threadsPerBlock-1)/threadsPerBlock);
    // Pointers to the device memory
    float *dev_a, *dev_b, *dev_partial_c;   // Partial C used to store the output of the multiplication step
    int *devArrsize;
    // Allocate memory to store our partial result on the CPU side
    float* partial_c = (float*)malloc(blocksPerGrid*sizeof(float));
    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, arrSize * sizeof(float));
    cudaMalloc((void**)&dev_b, arrSize * sizeof(float));
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&devArrsize, sizeof(int));
    // Copy inputs to the device
    cudaMemcpy(dev_a, a, arrSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arrSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devArrsize, &arrSize, sizeof(int), cudaMemcpyHostToDevice);
    // Run the kernel
    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c, devArrsize);
    // Copy our partial array back to the host
    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
    // Finish up the dot product on the CPU side
    float retVal = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        retVal += partial_c[i];
    }
    // Return our result
    return retVal;
}



}   // namespace cuda_ros_node