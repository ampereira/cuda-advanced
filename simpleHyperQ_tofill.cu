//
// This sample demonstrates how HyperQ allows supporting devices to avoid false
// dependencies between kernels in different streams.
//
// - Devices without HyperQ will run a maximum of two kernels at a time (one
//   kernel_A and one kernel_B).
// - Devices with HyperQ will run up to 32 kernels simultaneously.

#include <stdio.h>

const char *sSDKsample = "hyperQ";

// This subroutine does no real work but runs for at least the specified number
// of clock ticks.
__device__ void clock_block(clock_t *d_o, clock_t clock_count)
{
    clock_t start_clock = clock();

    clock_t clock_offset = 0;

    while (clock_offset < clock_count)
    {
        clock_offset = clock() - start_clock;
    }

    d_o[0] = clock_offset;
}

// We create two identical kernels calling clock_block(), we create two so that
// we can identify dependencies in the profile timeline ("kernel_B" is always
// dependent on "kernel_A" in the same stream).
__global__ void kernel_A(clock_t *d_o, clock_t clock_count)
{
    clock_block(d_o, clock_count);
}
__global__ void kernel_B(clock_t *d_o, clock_t clock_count)
{
    clock_block(d_o, clock_count);
}

// Single-warp reduction kernel (note: this is not optimized for simplicity)
__global__ void sum(clock_t *d_clocks, int N)
{
    __shared__ clock_t s_clocks[32];

    clock_t my_sum = 0;

    for (int i = threadIdx.x ; i < N ; i += blockDim.x)
    {
        my_sum += d_clocks[i];
    }

    s_clocks[threadIdx.x] = my_sum;
    __syncthreads();

    for (int i = warpSize / 2 ; i > 0 ; i /= 2)
    {
        if (threadIdx.x < i)
        {
            s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        d_clocks[0] = s_clocks[0];
    }
}


int main(int argc, char **argv)
{
    int nstreams = 32;          // One stream for each pair of kernels
    float kernel_time = 10;     // Time each kernel should run in ms
    float elapsed_time;
    int cuda_device = 0;

// *************************************************************************************
// *************************************************************************************
    printf("starting %s...\n", sSDKsample);

    // Get number of streams (if overridden on the command line)
        nstreams = 16;    }
// *************************************************************************************
// *************************************************************************************

    // Allocate host memory for the output (reduced to a single value)
    clock_t *a = 0;
    cudaMallocHost((void **)&a, sizeof(clock_t));

    // Allocate device memory for the output (one value for each kernel)
    clock_t *d_a = 0;

    // Allocate and initialise an array of stream handles


// *************************************************************************************
// *************************************************************************************
    // Create CUDA event handles
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Target time per kernel is kernel_time ms, clockRate is in KHz
    // Target number of clocks = target time * clock frequency

    // Start the clock
    cudaEventRecord(start_event, 0);
// *************************************************************************************
// *************************************************************************************

    // Queue pairs of {kernel_A, kernel_B} in separate streams

// *************************************************************************************
// *************************************************************************************
    // Stop the clock in stream 0 (i.e. all previous kernels will be complete)
    cudaEventRecord(stop_event, 0);
// *************************************************************************************
// *************************************************************************************

    // At this point the CPU has dispatched all work for the GPU and can
    // continue processing other tasks in parallel. In this sample we just want
    // to wait until all work is done so we use a blocking cudaMemcpy below.

    // Run the sum kernel and copy the result back to host


// *************************************************************************************
// *************************************************************************************
    // stop_event will have been recorded but including the synchronize here to
    // prevent copy/paste errors!
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

    printf("Expected time for serial execution of %d sets of kernels is between approx. %.3fs and %.3fs\n", nstreams, (nstreams + 1) * kernel_time / 1000.0f, 2 * nstreams *kernel_time / 1000.0f);
    printf("Expected time for fully concurrent execution of %d sets of kernels is approx. %.3fs\n", nstreams, 2 * kernel_time / 1000.0f);
    printf("Measured time for sample = %.3fs\n", elapsed_time / 1000.0f);

// *************************************************************************************
// *************************************************************************************

    // Release resources (including streams)


// *************************************************************************************
// *************************************************************************************
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(true);
// *************************************************************************************
// *************************************************************************************
}
