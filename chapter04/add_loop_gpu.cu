#include "../common/book.h"

#define N 10

__global__ void add(int* a, int* b, int* c)
{
    int tid = blockIdx.x;   // handle the data at this index
    if(tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void)
{
    int a[N], b[N], c[N]; // Arrays arrays to be created and stored on host
    int *dev_a, *dev_b, *dev_c; // Pointers to arrays to be created and stored on device

    // Allocate memory on GPU
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(int)));

    // Fill arrays 'a' and 'b' on host CPU
    for(int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i + i;
    }

    // Copy arrays 'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));

    add<<<N,1>>>(dev_a, dev_b, dev_c);

    // Copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));

    // Display the results
    for(int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free allocated memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}