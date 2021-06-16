#include "Visualization.cuh"
#include <iostream>

#define TARGET_HEIGHT 1000
#define TARGET_WIDTH 1500

Visualization::Visualization(float minX, float maxX, float minY, float maxY, float (*f)(float, float),
                             std::vector<std::vector<Particle>> posList) : minX(minX),
                                                                           minY(minY) {
    int iteration = 0;
    int frame = 0;


    std::cout << posList[0].size() << std::endl;
    setupScale(minX, maxX, minY, maxY);
    sf::RenderWindow window(sf::VideoMode(width, height), "Particle swarm Optimisation - visualization",
                            sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(120);
    float **map;
    float max;
    float min;
    sf::Image image;
    sf::Texture tex;
    sf::Sprite background;

    calculateFunctionValues(minX, minY, f, map, max, min);
    image.create(width, height, sf::Color(100, 0, 0));
    createMap(minX, minY, f, map, max, min, image);
    tex.loadFromImage(image);
    background.setTexture(tex);
    sf::CircleShape particle(3, 8);
    particle.setOrigin(3, 3);
    particle.setFillColor(sf::Color(100, 0, 0));

    while (window.isOpen()) {
        handleEvents(window);
        window.clear();
        window.draw(background);


        for (Particle p: posList[iteration]) {
            Position pixel = particleToPixel(p.current_position);
            particle.setPosition(pixel.x, pixel.y);
            window.draw(particle);
        }

        window.display();

        if (frame == 20) {
            frame = 0;
            iteration++;
        }
        if (iteration == posList.size()) iteration = posList.size() - 1;
        frame++;
    }
}

void Visualization::handleEvents(sf::RenderWindow &window) const {
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed)
            window.close();
    }
}

void Visualization::createMap(float minX, float minY, float (*f)(float, float), float **map, float max, float min,
                              sf::Image &image) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int val = (map[i][j] - min) * 255 / (max - min);
            image.setPixel(j, i, sf::Color(val, val, val));
        }
    }
}

void Visualization::calculateFunctionValues(float minX, float minY, float (*f)(float, float), float **&map, float &max,
                                            float &min) {
    map = new float *[height];
    max = f(minX, minY);
    min = f(minX, minY);
    for (int i = 0; i < height; i++)
        map[i] = new float[width];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            Position pixel(j, i);
            Position p = pixelToParticle(pixel);
            int y = (int) p.y;
            int x = (int) p.x;
            float val = f(x, y);
            map[i][j] = val;
            if (val > max) max = val;
            if (val < min) min = val;
        }
    }
}

void Visualization::setupScale(float minX, float maxX, float minY, float maxY) {
    double xScale = TARGET_WIDTH / (maxX - minX);
    double yScale = TARGET_HEIGHT / (maxY - minY);

    if (xScale < yScale) {
        scale = xScale;
        width = TARGET_WIDTH;
        height = (maxY - minY) * scale;
    } else {
        scale = yScale;
        height = TARGET_HEIGHT;
        width = (maxX - minX) * scale;
    }
}

Position Visualization::particleToPixel(Position particle) {
    return Position((particle.x - minX) * scale, (particle.y - minY) * scale);
}

Position Visualization::pixelToParticle(Position pixel) {
    return Position(pixel.x / scale + minX, pixel.y / scale + minY);
}
