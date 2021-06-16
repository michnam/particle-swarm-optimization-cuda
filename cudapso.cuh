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


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
std::vector<std::vector<Particle>> performPso();


#endif //PSO_CUDAPSO_CUH
