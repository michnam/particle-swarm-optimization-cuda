//
// Created by dedless on 24.05.2021.
//

#include "pso.cuh"

void hello() {
    hello_kernel<<<1,1>>>();
}


__global__ void hello_kernel() {
    printf("Hello From GPU\n");
}