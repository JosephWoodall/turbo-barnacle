using System;

// Product
abstract class Animal
{
    public abstract string Speak();
}

// Concrete Products
class Dog : Animal
{
    public override string Speak()
    {
        return "Woof!";
    }
}

class Cat : Animal
{
    public override string Speak()
    {
        return "Meow";
    }
}

// Creator
abstract class AnimalFactory
{
    public abstract Animal CreateAnimal();
}

// Concrete Creators
class DogFactory : AnimalFactory
{
    public override Animal CreateAnimal()
    {
        return new Dog();
    }
}

class CatFactory : AnimalFactory
{
    public override Animal CreateAnimal()
    {
        return new Cat();
    }
}

// Client code
class Client
{
    static void Main()
    {
        AnimalFactory factory;
        Animal animal;

        factory = new DogFactory();
        animal = factory.CreateAnimal();
        Console.WriteLine(animal.Speak());  // Output: "Woof!"

        factory = new CatFactory();
        animal = factory.CreateAnimal();
        Console.WriteLine(animal.Speak());  // Output: "Meow"
    }
}
