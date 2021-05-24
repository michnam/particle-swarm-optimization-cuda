#include <iostream>
#include <SFML/Graphics.hpp>
#include "pso.cuh"

int main() {
    std::cout << "Hello, World!" << std::endl;
    hello();


    sf::RenderWindow window(sf::VideoMode(200, 200), "SFML works!");
    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        cudaDeviceSynchronize();

        window.clear();
        window.draw(shape);
        window.display();
    }
    return 0;
}
