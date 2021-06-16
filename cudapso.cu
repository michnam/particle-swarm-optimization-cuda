#include <curand_kernel.h>
#include "cudapso.cuh"

#include "cudapso.cuh"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

__global__ void addKernel(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

const unsigned int N = 500;
const unsigned int ITERATIONS = 1000;
const float SEARCH_MIN = -1000.0f;
const float SEARCH_MAX = 1000.0f;
const float w = 0.9f;
const float c_ind = 1.0f;
const float c_team = 2.0f;

// return a random float between low and high
float randomFloat(float low, float high) {
    float range = high - low;
    float pct = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return low + pct * range;
}

// function to optimize
// TODO: We need to pass that from main
__device__ __host__ float calcValue(Position p) {
    return p.x * p.x + p.y * p.y;
}

// Initialize state for random numbers
__global__ void init_kernel(curandState *state, long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, state);
}

// Returns the index of the particle with the best position
__global__ void updateTeamBestIndex(Particle *d_particles,
                                    float *team_best_value,
                                    int *team_best_index, int N) {
    __shared__ float best_value;
    __shared__ int best_index;
    best_value = d_particles[0].best_value;
    best_index = 0;
    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        if (d_particles[idx].best_value < best_value) {
            best_value = d_particles[idx].best_value;
            best_index = idx;
            __syncthreads();
        }
    }
    *team_best_value = best_value;
    *team_best_index = best_index;
}

__global__ void updateVelocity(Particle *d_particles, int *team_best_index,
                               float w, float c_ind, float c_team, int N,
                               curandState *state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float best_x, best_y;
    best_x = d_particles[*team_best_index].best_position.x;
    best_y = d_particles[*team_best_index].best_position.y;
    __syncthreads();

    if (idx < N) {
        float r_ind = curand_uniform(state);
        float r_team = curand_uniform(state);
        d_particles[idx].velocity.x =
                w * d_particles[idx].velocity.x +
                r_ind * c_ind *
                (d_particles[idx].best_position.x -
                 d_particles[idx].current_position.x) +
                r_team * c_team * (best_x - d_particles[idx].current_position.x);
        d_particles[idx].velocity.y =
                w * d_particles[idx].velocity.y +
                r_ind * c_ind *
                (d_particles[idx].best_position.y -
                 d_particles[idx].current_position.y) +
                r_team * c_team * (best_y - d_particles[idx].current_position.y);
    }
}

__global__ void updatePosition(Particle *d_particles, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        d_particles[idx].current_position += d_particles[idx].velocity;
        float newValue = calcValue(d_particles[idx].current_position);
        if (newValue < d_particles[idx].best_value) {
            d_particles[idx].best_value = newValue;
            d_particles[idx].best_position = d_particles[idx].current_position;
        }
    }
}

vector<vector<Particle>> performPso() {
    cudaError_t cudaStatus;
    // for timing
    long start = clock();
    std::vector<std::vector<Particle>> positionHistory;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
        goto Error;
    }
    // Random seed for cpu
    unsigned seed = time(nullptr);
    srand(seed);

    curandState *state;
    cudaStatus = cudaMalloc(&state, sizeof(curandState));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
        goto Error;
    }
    init_kernel<<<1, 1>>>(state, clock());

    // Initialize particles
    Particle *particles;
    size_t particleSize = sizeof(Particle) * N;

    // initialize variables for team best
    int *team_best_index;
    float *team_best_value;

    // Allocate particles in unified memory
    cudaStatus = cudaMallocManaged(&particles, particleSize);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMallocManaged failed!";
        goto Error;
    }
    cudaStatus = cudaMallocManaged(&team_best_index, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMallocManaged failed!";
        goto Error;
    }

    // Allocate team_best_value for gpu only
    cudaStatus = cudaMalloc(&team_best_value, sizeof(float));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
        goto Error;
    }

    // Prefetch data to the GPU
    int device = 0;
    cudaStatus = cudaGetDevice(&device);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaGetDevice failed!";
        goto Error;
    }
    cudaMemPrefetchAsync(team_best_index, sizeof(int), device,
                         0);  // ptr, size_t, device, stream

    // Memory hints
    cudaMemAdvise(particles, particleSize, cudaMemAdviseSetPreferredLocation,
                  cudaCpuDeviceId);  // start on cpu

    //  Initialize particles on host
    for (int i = 0; i < N; i++) {
        // Random starting position
        particles[i].current_position.x = randomFloat(SEARCH_MIN, SEARCH_MAX);
        particles[i].current_position.y = randomFloat(SEARCH_MIN, SEARCH_MAX);
        particles[i].best_position.x = particles[i].current_position.x;
        particles[i].best_position.y = particles[i].current_position.y;
        particles[i].best_value = calcValue(particles[i].best_position);
        // Random starting velocity
        particles[i].velocity.x = randomFloat(SEARCH_MIN, SEARCH_MAX);
        particles[i].velocity.y = randomFloat(SEARCH_MIN, SEARCH_MAX);
    }

    // Prefetch particles to gpu
    cudaMemPrefetchAsync(particles, particleSize, device, 0);

    // Initialize team best index and value
    updateTeamBestIndex<<<1, 1>>>(particles, team_best_value, team_best_index, N);

    // assign thread and blockcount
    int blockSize = 32;
    int gridSize = (N + blockSize - 1) / blockSize;



    // For i in interations
    for (int i = 0; i < ITERATIONS; i++) {

        updateVelocity<<<gridSize, blockSize>>>(particles, team_best_index, w,
                                                c_ind, c_team, N, state);
        updatePosition<<<gridSize, blockSize>>>(particles, N);
        updateTeamBestIndex<<<gridSize, blockSize>>>(particles, team_best_value,
                                                     team_best_index, N);
        Particle *particlesGPU = new Particle[particleSize];
        cudaMemcpy(particles, particlesGPU, particleSize, cudaMemcpyHostToDevice);
        std::vector<Particle> positionList(particlesGPU, particlesGPU + N);
        positionHistory.push_back(positionList);
    }

    // Wait for gpu to finish computation
    cudaDeviceSynchronize();

    // Prefetch particles and best index back
    cudaMemPrefetchAsync(particles, particleSize, cudaCpuDeviceId);
    cudaMemPrefetchAsync(team_best_index, sizeof(int), cudaCpuDeviceId);

    // Stop clock
    long stop = clock();
    long elapsed = (stop - start) * 1000 / CLOCKS_PER_SEC;

    cout << "Ending Best: " << endl;
    cout << "Team best value: " << particles[*team_best_index].best_value << endl;
    cout << "Team best position: " << particles[*team_best_index].best_position.x
         << ", " << particles[*team_best_index].best_position.y << endl;

    cout << "Run time: " << elapsed << "ms" << endl;

    Error:
    cudaFree(particles);
    cudaFree(team_best_index);
    cudaFree(team_best_value);
    cudaFree(state);
    return positionHistory;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr,
                "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n",
                cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr,
                "cudaDeviceSynchronize returned error code %d after launching "
                "addKernel!\n",
                cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
