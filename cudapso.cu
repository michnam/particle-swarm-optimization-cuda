#include <curand_kernel.h>
#include "cudapso.cuh"

#include <cmath>
#include <ctime>
#include <iostream>
#include <stdlib.h>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
using namespace std;

const float w = 0.1f;
const float c_ind = 1.0f;
const float c_team = 2.0f;

// return a random float between low and high
float randomFloat(float low, float high) {
    float range = high - low;
    float pct = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return low + pct * range;
}

__device__ __host__ float exampleFunA_GPU(Position p) {
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
    float best_value = d_particles[0].best_value;
    int best_index = 0;

    for (int i = 0; i < N; i++) {
        if (d_particles[i].best_value < best_value) {
            best_value = d_particles[i].best_value;
            best_index = i;
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
        float newValue;
        switch (fun) {
            case FuncA: {
                newValue = exampleFunA_GPU(d_particles[idx].current_position);
                break;
            }
            case FuncB: {
                newValue = exampleFunB_GPU(d_particles[idx].current_position);
                break;
            }
            case FuncC: {
                newValue = exampleFunC_GPU(d_particles[idx].current_position);
                break;
            }
        }
        if (d_particles[idx].current_position.x < min_x) d_particles[idx].current_position.x = min_x;
        if (d_particles[idx].current_position.x > max_x) d_particles[idx].current_position.x = max_x;
        if (d_particles[idx].current_position.y < min_y) d_particles[idx].current_position.y = min_y;
        if (d_particles[idx].current_position.y > max_y) d_particles[idx].current_position.y = max_y;
        if (newValue < d_particles[idx].best_value) {
            d_particles[idx].best_value = newValue;
            d_particles[idx].best_position = d_particles[idx].current_position;
        }
    }


}

vector<vector<Particle>> performPso(FunctionType fun, int min_x, int max_x, int min_y, int max_y, int particles_num, int iterations) {
    int search_min = min(min_x, min_y);
    int search_max = min(max_x, max_y);
    std::vector<std::vector<Particle>> positionHistory;
    cudaSetDevice(0);
    unsigned seed = time(nullptr);
    srand(seed);

    curandState *state;
    cudaMalloc(&state, sizeof(curandState));
    init_kernel<<<1, 1>>>(state, clock());

    // Initialize particles
    Particle *particles;
    size_t particleSize = sizeof(Particle) * N;

    // initialize variables for team best
    int *team_best_index;
    float *team_best_value;

    particles = new Particle[particles_num];
    cudaMalloc(&d_particles, particleSize);
    cudaMalloc(&d_team_best_index, sizeof(int));
    cudaMalloc(&team_best_value, sizeof(float));

    //  Initialize particles on host
    for (int i = 0; i < N; i++) {
        // Random starting position
        particles[i].current_position.x = randomFloat(SEARCH_MIN, SEARCH_MAX);
        particles[i].current_position.y = randomFloat(SEARCH_MIN, SEARCH_MAX);
        particles[i].best_position.x = particles[i].current_position.x;
        particles[i].best_position.y = particles[i].current_position.y;
        particles[i].best_value = calcValue(particles[i].best_position);
        // Random starting velocity
        particles[i].velocity.x = randomFloat(search_min / 1000, search_max / 1000);
        particles[i].velocity.y = randomFloat(search_min / 1000, search_max / 1000);
    }

    // Prefetch particles to gpu
    cudaMemPrefetchAsync(particles, particleSize, device, 0);

    // Initialize team best index and value
    updateTeamBestIndex<<<1, 1>>>(d_particles, team_best_value, d_team_best_index, particles_num);

    int blockSize = 32;
    int gridSize = (N + blockSize - 1) / blockSize;



    long totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        long loopStart = clock();
        updateVelocity<<<gridSize, blockSize>>>(d_particles, d_team_best_index, w,
                                                c_ind, c_team, particles_num, state);
        updatePosition<<<gridSize, blockSize>>>(d_particles, particles_num, min_x, max_x, min_y, max_y, fun);
        updateTeamBestIndex<<<1, 1>>>(d_particles, team_best_value, d_team_best_index, particles_num);
        long loopEnd = clock();
        totalTime += (loopEnd - loopStart) * 1000 / CLOCKS_PER_SEC;
        cudaMemcpy(particlesGPU, d_particles, particleSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(&team_best, team_best_value, sizeof(float), cudaMemcpyDeviceToHost);
        std::vector<Particle> positionList(particlesGPU, particlesGPU + particles_num);
        positionHistory.push_back(positionList);
    }

    // Wait for gpu to finish computation
    cudaDeviceSynchronize();

    cout << "Team best value: " << particles[team_best_index].best_value << endl;
    cout << "Team best position: " << particles[team_best_index].best_position.x
         << ", " << particles[team_best_index].best_position.y << endl;

    cout << "Run time: " << totalTime << "ms" << endl;

    delete[] particlesGPU;
    cudaFree(d_particles);
    cudaFree(d_team_best_index);
    cudaFree(team_best_value);
    cudaFree(state);
    return positionHistory;
}

#pragma clang diagnostic pop