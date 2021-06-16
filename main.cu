#pragma once

#include <iostream>
#include <cmath>
#include <ctime>
#include <SFML/Graphics.hpp>
#include "pso.cuh"
#include "Visualization.cuh"


float f1(float x, float y) {
    return 0.01 * (x * x + y * y) - 5;
}

float minX = -100;
float maxX = 300;
float minY = -50;
float maxY = 200;

int size = 500;

std::vector<Particle> randomPositions(int n) {
    std::vector<Particle> p;
    for (int i = 0; i < n; i++) {
        Particle particle;
        particle.current_position.x = (rand() % (int)(maxX - minX)) + minX;
        particle.current_position.y = (rand() % (int)(maxY - minY)) + minY;
        particle.velocity.x = rand() % 4 - 2;
        particle.velocity.y = rand() % 4 - 2;

        p.push_back(particle);
    }
    return p;
}


int main() {
    unsigned seed = time(nullptr);
    srand(seed);
    helloGPU();


    std::vector<std::vector<Particle>> posList;

    for (int i =0;i< 120; i++)
    {
        std::vector<Particle> pos = randomPositions(size);
        posList.push_back(pos);
    }

    Visualization v(minX, maxX, minY, maxY, f1, posList);


    return 0;
}


