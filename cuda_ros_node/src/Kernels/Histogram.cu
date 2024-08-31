#include "cuda.h"
#include <stdio.h>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)



  namespace cuda_ros_node {

  __global__ void histogram_kernel(unsigned char* buffer, std::size_t bufferSize, unsigned int* histogram)
  {
    // Allocate and zero a shared memory buffer to hold each block's intermediate histogram
    __shared__ unsigned int localHistogram[256];
    localHistogram[threadIdx.x] = 0;
    __syncthreads();  // Ensure every thread's write has completed before progressing

    // Determine the index and offset for this thread
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int offset = blockDim.x * gridDim.x;
    // Walk through the array incrementing the proper bin in the histogram
    while (i < bufferSize)
    {
      atomicAdd(&localHistogram[buffer[i]], 1);
      i += offset;
    }
    // Merge each thread's local histogram into the global histogram
    __syncthreads();
    atomicAdd(&histogram[threadIdx.x], localHistogram[threadIdx.x]);
  }

  void compute_histogram(unsigned char* inputBuffer, std::size_t inputBufferSize, unsigned int* histogram)
  {
    unsigned char* dev_buffer;
    unsigned int* dev_histogram;

    checkCudaErrors(cudaMalloc((void**)&dev_buffer, inputBufferSize));
    checkCudaErrors(cudaMemcpy(dev_buffer, inputBuffer, inputBufferSize, cudaMemcpyHostToDevice));

    // Initialize our histogram on the GPU
    checkCudaErrors(cudaMalloc((void**)&dev_histogram, 256 * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(dev_histogram, 0, 256 * sizeof(unsigned int)));

    // Dynamically size our launch based on the current hardware platform
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;

    // Run the kernel
    histogram_kernel<<<blocks*2, 256>>>(dev_buffer, inputBufferSize, dev_histogram);

    // Copy the histogram back to the host
    checkCudaErrors(cudaMemcpy(histogram, dev_histogram, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Clean up
    checkCudaErrors(cudaFree(dev_buffer));
    checkCudaErrors(cudaFree(dev_histogram));
  }




  } // namespace cuda_ros_node