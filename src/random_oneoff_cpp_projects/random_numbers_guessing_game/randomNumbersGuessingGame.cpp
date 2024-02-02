#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main()
{
    srand(time(0)); // seed random number generator

    int maxNumber = 10;
    int secretNumber = rand() % maxNumber; // random number between 1 and 100
    int guessCount = 0;
    int maxGuesses = 5;
    int guess; // user's guess

    cout << "Welcome to the Random Numbers Guessing Game!" << endl;

    while (guessCount < maxGuesses && guess != secretNumber)
    {
        cout << "Enter a guess between 0 and " << maxNumber << endl;
        cin >> guess;

        guessCount++;

        if (guess == secretNumber)
        {
            cout << "Congratulations! You guessed the number in " << guessCount << " tries!" << endl;
        }
        else if (guess < secretNumber)
        {
            cout << "Too low! Try again." << endl;
            cout << "You have " << maxGuesses - guessCount << " guesses left." << endl;
        }
        else
        {
            cout << "Too high! Try again." << endl;
            cout << "You have " << maxGuesses - guessCount << " guesses left." << endl;
        }

        if (guessCount == maxGuesses)
        {
            cout << "Sorry, you've reached the maximum number of guesses. The secret number was: " << secretNumber << endl;
        }
    }
    return 0;
}