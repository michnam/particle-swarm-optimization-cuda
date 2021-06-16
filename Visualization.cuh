#ifndef PSO_VISUALIZATION_CUH
#define PSO_VISUALIZATION_CUH
#pragma once

#include <SFML/Graphics.hpp>
#include "cudapso.cuh"


/**
 * Class used for visualization of particles positions after PSO optimisation
 */
class Visualization {
private:
    int width;         /**< width of screen */
    int height;        /**< height of screen */
    double minX;       /**< minimal X bound */
    double minY;       /**< minimal Y bound */
    double scale;      /**< scale of screen to X / Y */

public:

    /**
     * Default constructor of class
     * @param minX minimal X bound
     * @param maxX max X bound
     * @param minY minimal Y bound
     * @param maxY max Y bound
     * @param f function to optimise
     * @param posList list of Particles for every iteration
     */
    Visualization(float minX, float maxX, float minY, float maxY, float (*f)(float, float),
                  std::vector<std::vector<Particle>> posList);

    /**
     * Changes particle position to position on screen
     * @param particle
     * @return particle witch changed position
     */
    Position particleToPixel(Position particle);

    /**
     * Changes particle pixel position on screen to x / y in function
     * @param particle
     * @return particle witch changed position
     */
    Position pixelToParticle(Position pixel);

    /**
     * Calculate scale for window
     * @param minX minX
     * @param maxX maxX
     * @param minY minY
     * @param maxY maxY
     */
    void setupScale(float minX, float maxX, float minY, float maxY);

    /**
     * Calculate value for every pixel in map
     * @param minX
     * @param minY
     * @param f
     * @param map
     * @param max
     * @param min
     */
    void
    calculateFunctionValues(float minX, float minY, float (*f)(float, float), float **&map, float &max, float &min);

    /**
     * create background for visualization, the darker, the lower value of function
     * @param minX
     * @param minY
     * @param f
     * @param map
     * @param max
     * @param min
     * @param image
     */
    void
    createMap(float minX, float minY, float (*f)(float, float), float **map, float max, float min, sf::Image &image);

    /**
     * handle quitting of application
     * @param window
     */
    void handleEvents(sf::RenderWindow &window) const;
};


#endif //PSO_VISUALIZATION_CUH
