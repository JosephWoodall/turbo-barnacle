class SubsystemA
{
    public string OperationA()
    {
        return "Subsystem A operation";
    }
}

class SubsystemB
{
    public string OperationB()
    {
        return "Subsystem B operation";
    }
}

class Facade
{
    private readonly SubsystemA _subsystemA;
    private readonly SubsystemB _subsystemB;

    public Facade(SubsystemA subsystemA, SubsystemB subsystemB)
    {
        _subsystemA = subsystemA;
        _subsystemB = subsystemB;
    }

    public string Operation()
    {
        var results = new List<string>();
        results.Add("Facade initializes subsystems:");
        results.Add(_subsystemA.OperationA());
        results.Add(_subsystemB.OperationB());
        return string.Join("\n", results);
    }
}

/*
// Usage
var subsystemA = new SubsystemA();
var subsystemB = new SubsystemB();
var facade = new Facade(subsystemA, subsystemB);
Console.WriteLine(facade.Operation());  // Output: Facade initializes subsystems:\nSubsystem A operation\nSubsystem B operation
*/