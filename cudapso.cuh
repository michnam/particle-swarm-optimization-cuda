#ifndef PSO_CUDAPSO_CUH
#define PSO_CUDAPSO_CUH

#include <iostream>

void helloGPU();
__global__ void hello_kernel();

struct Position {
    float x, y;

    Position() {}

    Position(float x, float y);

    void operator+=(const Position &a);

    void operator=(const Position &a);
};

struct Particle {
    Particle() {};

    Position best_position;
    Position current_position;
    Position velocity;
    float best_value;
};



#endif //PSO_CUDAPSO_CUH
