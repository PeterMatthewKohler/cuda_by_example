#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace cuda_ros_node {
__global__ void add(int a, int b, int* c)
{
    *c = a + b;
}

void cudaAdd(int* c)
{
    //int a = 5, b = 12; // Host copies of a, b, c
    int* dev_c; // Device copies of a, b, c
    // Alloc space for device copies of c
    cudaMalloc((void **)&dev_c, sizeof(int));
    // Copy inputs to device
    // cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    // Launch add() kernel
    add<<<1,1>>>(5, 12, dev_c);
    // Copy results back to the host
    cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    // Cleanup allocated memory
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
        printf("  Threads in warp: %d\n", prop.maxThreadsPerBlock);
        printf("  Max thread dimensions: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}
} // namespace cuda_ros_node