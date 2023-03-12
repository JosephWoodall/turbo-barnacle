// Component
abstract class Component
{
    public abstract string Operation();
}

// Concrete Component
class ConcreteComponent : Component
{
    public override string Operation()
    {
        return "ConcreteComponent";
    }
}

// Decorator
abstract class Decorator : Component
{
    protected Component _component;

    public Decorator(Component component)
    {
        _component = component;
    }

    public override string Operation()
    {
        return _component.Operation();
    }
}

// Concrete Decorator A
class ConcreteDecoratorA : Decorator
{
    public ConcreteDecoratorA(Component component) : base(component) { }

    public override string Operation()
    {
        return $"ConcreteDecoratorA({base.Operation()})";
    }
}

// Concrete Decorator B
class ConcreteDecoratorB : Decorator
{
    public ConcreteDecoratorB(Component component) : base(component) { }

    public override string Operation()
    {
        return $"ConcreteDecoratorB({base.Operation()})";
    }
}

/*
// Usage
Component component = new ConcreteComponent();
Decorator decorator1 = new ConcreteDecoratorA(component);
Decorator decorator2 = new ConcreteDecoratorB(decorator1);
Console.WriteLine(decorator2.Operation());  // Output: ConcreteDecoratorB(ConcreteDecoratorA(ConcreteComponent))
*/