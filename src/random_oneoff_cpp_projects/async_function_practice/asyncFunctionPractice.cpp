#include <iostream>
#include <vector>
#include <random>
#include <string>
#include "Actor.h"

#include <cstdlib>
#include <ctime>
#include <future>

// Asynchronous function to randomly change the mood of an actor
void changeMoodAsync(Actor &actor)
{
    // Seed the random number generator
    std::srand(std::time(nullptr));

    // Create a future to hold the result of the asynchronous operation
    std::future<void> future = std::async(std::launch::async, [&actor]()
                                          {
        // Generate a random mood value between 1 and 10
        int randomMood = std::rand() % 10 + 1;

        // Update the actor's mood
        actor.mood = randomMood; });

    // Wait for the asynchronous operation to complete
    future.wait();
}

int generateRandomInteger()
{
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;
    // Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen(rd());
    // Define the range for the random integer
    std::uniform_int_distribution<> distrib(1, 6);
    // Generate random integer
    int random_integer = distrib(gen);

    return random_integer;
}

int main()
{

    while (true)
    {

        // Create a vector to store the actors
        std::vector<Actor> actors;
        int actor_count = 5;

        // Push actor objects to the vector
        for (int i = 0; i < actor_count; i++)
        {
            actors.push_back(Actor());
        }
        std::cout << "There are " << actors.size() << " actors in the simulation." << std::endl;
        for (auto &actor : actors)
        {
            actor.name = "Actor";
            actor.mood = generateRandomInteger();
            actor.lastTimeAte = generateRandomInteger();
            actor.endOfDayTime = generateRandomInteger();

            std::cout << "Actor: " << actor.name << " Mood: " << actor.mood << " Last Time Ate: " << actor.lastTimeAte << " End of Day Time: " << actor.endOfDayTime << std::endl;
        }

        // Perform simulation operations here
        // Change the mood of each actor asynchronously
        for (auto &actor : actors)
        {
            changeMoodAsync(actor);
            std::cout << "There's been a change!" << std::endl;
            std::cout << "Actor: " << actor.name << " Mood: " << actor.mood << " Last Time Ate: " << actor.lastTimeAte << " End of Day Time: " << actor.endOfDayTime << std::endl;
        }
    }
    return 0;
}