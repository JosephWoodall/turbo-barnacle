using System;
using System.Collections.Generic;

class Flyweight
{
    private readonly string _sharedState;

    public Flyweight(string sharedState)
    {
        _sharedState = sharedState;
    }

    public string Operation(string uniqueState)
    {
        return $"Flyweight: Shared '{_sharedState}' and unique '{uniqueState}'";
    }
}

class FlyweightFactory
{
    private readonly Dictionary<string, Flyweight> _flyweights = new Dictionary<string, Flyweight>();

    public Flyweight GetFlyweight(string sharedState)
    {
        if (!_flyweights.ContainsKey(sharedState))
        {
            Console.WriteLine("FlyweightFactory: Can't find a flyweight, creating new one.");
            _flyweights[sharedState] = new Flyweight(sharedState);
        }
        else
        {
            Console.WriteLine("FlyweightFactory: Reusing existing flyweight.");
        }
        return _flyweights[sharedState];
    }
}

/*
// Usage
var flyweightFactory = new FlyweightFactory();
var flyweight1 = flyweightFactory.GetFlyweight("shared_state_1");
Console.WriteLine(flyweight1.Operation("unique_state_1")); // Output: Flyweight: Shared 'shared_state_1' and unique 'unique_state_1'

var flyweight2 = flyweightFactory.GetFlyweight("shared_state_2");
Console.WriteLine(flyweight2.Operation("unique_state_2")); // Output: Flyweight: Shared 'shared_state_2' and unique 'unique_state_2'

var flyweight3 = flyweightFactory.GetFlyweight("shared_state_1");
Console.WriteLine(flyweight3.Operation("unique_state_3")); // Output: Flyweight: Shared 'shared_state_1' and unique 'unique_state_3'
*/