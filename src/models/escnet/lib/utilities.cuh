#ifndef _MY_UTILITIES
#define _MY_UTILITIES

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__device__ scalar_t AtomicAdd(scalar_t* address, scalar_t val)
{
    return atomicAdd(address, val);
}


// Lifted from https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicadd
template <>
__device__ double AtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}


#endif