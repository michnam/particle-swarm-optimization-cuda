#include "pso.cuh"

void helloGPU() {
    hello_kernel<<<1,1>>>();
}


__global__ void hello_kernel() {
    printf("Hello From GPU\n");
}