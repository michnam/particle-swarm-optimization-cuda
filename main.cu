#pragma once

#include <iostream>
#include <cmath>
#include <ctime>
#include <SFML/Graphics.hpp>
#include "cudapso.cuh"
#include "Visualization.cuh"

using namespace std;

__device__ __host__ float calcValue(float x, float y) {
    return x * x + y * y;
}

const float SEARCH_MIN = -1000.0f;
const float SEARCH_MAX = 1000.0f;

double exampleFunA(double x, double y) {
    printf("Hello from example function A");
    return 1.0 / (x + y);
}

double exampleFunB(double x, double y) {
    printf("Hello from example function B");
    return pow(x, y);
}

double exampleFunC(double x, double y) {
    printf("Hello from example function C");
    return y * sin(x) + x * cos(y);
}

void userHello() {
    printf("Hi! Please select example function:\n");
    printf("1: 1/(x + y)\n");
    printf("2: x^y\n");
    printf("3: y * sin x + x * cos y\n");
    printf("Anything else: quit\n");
    printf("Your choice: ");
}

int main() {
    const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};
    char userInput;
    bool isRunning = true;
    cudaError_t cudaStatus;

    while (isRunning) {
        userHello();
        scanf(" %c", &userInput);

        switch (userInput) {
            case '1':
            case '2':
            case '3': {
                // Add vectors in parallel.
                std::vector<std::vector<Particle>> positions = performPso();

                Visualization(SEARCH_MIN, SEARCH_MAX, SEARCH_MIN, SEARCH_MAX, calcValue, positions);


                //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0],
                //       c[1], c[2], c[3], c[4]);

                break;
            }

            default:
                isRunning = false;
        }
    }

    // cudaDeviceReset must be called before exiting in order for profiling
    // and tracing tools such as Nsight and Visual Profiler to show complete
    // traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


