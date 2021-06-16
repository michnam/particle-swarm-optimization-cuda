#ifndef PSO_VISUALIZATION_CUH
#define PSO_VISUALIZATION_CUH
#pragma once

#include <SFML/Graphics.hpp>


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

class Visualization {
private:
    int width;
    int height;
    double minX;
    double maxX;
    double minY;
    double maxY;
    double scale;

public:
    Visualization(float minX, float maxX, float minY, float maxY, float (*f)(float, float),
                  std::vector<std::vector<Particle>>);

    Position particleToPixel(Position particle);

    Position pixelToParticle(Position pixel);

    void setupScale(float minX, float maxX, float minY, float maxY);

    void
    calculateFunctionValues(float minX, float minY, float (*f)(float, float), float **&map, float &max, float &min);

    void
    createMap(float minX, float minY, float (*f)(float, float), float **map, float max, float min, sf::Image &image);

    void handleEvents(sf::RenderWindow &window) const;
};


#endif //PSO_VISUALIZATION_CUH
