#include "cudapso.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void helloGPU() {
    hello_kernel<<<1, 1>>>();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void hello_kernel() {
    printf("Hello GPU\n");
}


void Position::operator=(const Position &a) {
    x = a.x;
    y = a.y;
}

void Position::operator+=(const Position &a) {
    x = x + a.x;
    y = y + a.y;
}

Position::Position(float x, float y) : x(x), y(y) {}
