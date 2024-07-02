#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace cuda_ros_node {

// Arbitrarily define some fixed number of threads we allow per block
// to ensure we never attempt to call more than exist
int maxThreadsAllowedPerBlock = 128;

__global__ void add(int* a, int* b, int* c, int* arrSize)
{
    // Perform operation on this index
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);  // Linearize the 2-D index space
    while(tid < *arrSize)  // If our index overshoots the end of the array, don't perform a calculation
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

void cudaAdd(int* a, int* b, int* c, int arrSize)
{
    // Pointers to the memory we will allocate on the device
    int *dev_a, *dev_b, *dev_c, *dev_ArrSize;

    // Allocate the memory on the device
    cudaMalloc((void **)&dev_a, arrSize * sizeof(int));
    cudaMalloc((void **)&dev_b, arrSize * sizeof(int));
    cudaMalloc((void **)&dev_c, arrSize * sizeof(int));
    cudaMalloc((void **)&dev_ArrSize, sizeof(int));
    // Copy inputs to device
    cudaMemcpy(dev_a, a, arrSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arrSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ArrSize, &arrSize, sizeof(int), cudaMemcpyHostToDevice);
    // Launch add() kernel
    add<<<maxThreadsAllowedPerBlock, maxThreadsAllowedPerBlock>>>(dev_a, dev_b, dev_c, dev_ArrSize);
    // Copy results back to the host
    cudaMemcpy(c, dev_c, arrSize * sizeof(int), cudaMemcpyDeviceToHost);
    // Cleanup allocated memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return;
}

void cudaPrintDeviceProperties()
{
    cudaDeviceProp prop;

    int count;
    cudaGetDeviceCount(&count);
    printf("Number of CUDA devices: %d\n", count);
    for(int i=0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d\n", i);
        printf("  Name: %s\n", prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %lu\n", prop.totalGlobalMem);
        printf("  Shared memory per block: %lu\n", prop.sharedMemPerBlock);
        printf("  Registers per block: %d\n", prop.regsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max thread dimensions: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}
} // namespace cuda_ros_node