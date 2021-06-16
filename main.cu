#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <SFML/Graphics.hpp>
#include "cudapso.cuh"
#include "Visualization.cuh"

using namespace std;

__device__ __host__ float calcValue(float x, float y) {
    return x * x + y * y;
}

float exampleFunA(float x, float y) {
    return x * x + y * y;
}

float exampleFunB(float x, float y) {
    float new_x = x + 10;
    return sin(new_x + y) + pow(new_x - y, 2) - 1.5 * new_x + 2.5 * y + 1 + new_x * y;
}

float exampleFunC(float x, float y) {
    return pow(x + 2 * y - 7, 2) + pow(2 * x + y - 5, 2);
    //return x * sin(sqrt(abs(x))) + y * sin(sqrt(abs(y)));
}

void userHello() {
    printf("Hi! Please select example function:\n");
    printf("1: x^2 + y^2\n");
    printf("2: sin(x+y+10) + (x-y+10)^2 - 1.5*(x+10) +2.5*y + 1 + (x+10)*y\n");
    printf("3: (x + 2 * y - 7) ^ 2 + (2 * x + y - 5) ^ 2\n");
    printf("Your choice: ");
}

int main() {
    char userInput;
    userHello();
    scanf(" %c", &userInput);

    float minX = -100;
    float minY = -100;
    float maxX = 100;
    float maxY = 100;
    int iterations = 1000;
    int particles_num = 500;

    cout << "Iterations: ";
    cin >> iterations;
    cout << "Particles: ";
    cin >> particles_num;
    cout << "minX: ";
    cin >> minX;
    cout << "minY: ";
    cin >> minY;
    cout << "maxX: ";
    cin >> maxX;
    cout << "maxY: ";
    cin >> maxY;

    switch (userInput) {
        case '1': {
            vector<vector<Particle>> positions = performPso(FuncA, minX, maxX, minY, maxY, particles_num, iterations);
            Visualization(minX, maxX, minY, maxY, exampleFunA, positions);
            break;
        }
        case '2': {
            vector<vector<Particle>> positions = performPso(FuncB, minX, maxX, minY, maxY, particles_num, iterations);
            Visualization(minX, maxX, minY, maxY, exampleFunB, positions);
            break;
        }
        case '3': {
            vector<vector<Particle>> positions = performPso(FuncC, minX, maxX, minY, maxY, particles_num, iterations);
            Visualization(minX, maxX, minY, maxY, exampleFunC, positions);
            break;
        }
    }

    return 0;
}


