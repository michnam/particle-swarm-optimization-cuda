#ifndef PSO_CUDAPSO_CUH
#define PSO_CUDAPSO_CUH

#include <iostream>
#include <vector>
#include <curand_kernel.h>

/**
 * Position struct contains x and y coordinates
 */
struct Position {
    /**
     * Position struct constructor
     * @param new_x -  x coordinate of particle position
     * @param new_y - y coordinate of particle position
     */
    Position(float new_x, float new_y) {
        x = new_x;
        y = new_y;
    }

    /**
     * Default constructor
     */
    Position() {

    }

    float x; /**< X coordinate of the particle position */
    float y; /**< Y coordinate of the particle position */

    /**
     *
     * @param a - number that we are adding to our particle position
     */
    __device__ __host__ void operator+=(const Position &a) {
        x = x + a.x;
        y = y + a.y;
    }

    __device__ __host__ void operator=(const Position &a) {
        x = a.x;
        y = a.y;
    }
};

// Particle struct has current location, best location and velocity
struct Particle {
    Particle() {};
    Position best_position;
    Position current_position;
    Position velocity;
    float best_value;
};

enum FunctionType {
    FuncA,
    FuncB,
    FuncC
};

/**
 * Kernel initializing state for random numbers
 * @param state - state to initialize
 * @param seed  - seed with which state is initialized
 */
__global__ void init_kernel(curandState *state, long seed);

/**
 * Function returning index of the particle with the best position and updating team best position coordinates
 * @param d_particles - all particles on the device
 * @param team_best_value - best value achieved by particles yet
 * @param team_best_index - index of the particle that achieved the best value
 * @param N - number of particles
 */
__global__ void updateTeamBestIndex(Particle *d_particles,
                                    float *team_best_value,
                                    int *team_best_index, int N);

/**
 * Updates the velocity of each particle
 * @param d_particles - all the particles on the device
 * @param team_best_index - index of the particle that achieved the best value
 * @param w - inertia of the particle (how fast it will move)
 * @param c_ind - willingness of the particle to move to the best position known to this particle
 * @param c_team - willingness of the particle to move to the best position known by team
 * @param N - number of particles
 * @param state - state used to get uniformly distributed float between 0.0 and 1.0
 */
__global__ void updateVelocity(Particle *d_particles, int *team_best_index,
                               float w, float c_ind, float c_team, int N,
                               curandState *state);
/**
 * Updates the position of the particles within boundaries
 * @param d_particles - all the particles on the device
 * @param N - number of particles
 * @param min_x - minimum x boundary
 * @param max_x - maximum x boundary
 * @param min_y - minimum y boundary
 * @param max_y - maximum y boundary
 * @param fun - function in which we are looking for minimum
 */
__global__ void
updatePosition(Particle *d_particles, int N, int min_x, int max_x, int min_y, int max_y, FunctionType fun);

/**
 * Main function performing PSO algorithm and returning particles positions over time to visualize the process
 * @param fun - function in which we are looking for minimum
 * @param min_x - minimum x boundary
 * @param max_x - maximum x boundary
 * @param min_y - minimum y boundary
 * @param max_y - maximum y boundary
 * @param particles_num - number of particles
 * @param iterations - number of iterations
 * @return vector containing particles over time (each iteration) to pass to visualization method
 */
std::vector<std::vector<Particle>>
performPso(FunctionType fun, int min_x, int max_x, int min_y, int max_y, int particles_num, int iterations);


#endif //PSO_CUDAPSO_CUH
