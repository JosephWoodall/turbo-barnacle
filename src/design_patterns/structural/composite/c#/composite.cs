// Component
abstract class Component
{
    public abstract string Operation();
}

// Leaf
class Leaf : Component
{
    public override string Operation()
    {
        return "Leaf";
    }
}

// Composite
class Composite : Component
{
    private List<Component> _children = new List<Component>();

    public void Add(Component component)
    {
        _children.Add(component);
    }

    public void Remove(Component component)
    {
        _children.Remove(component);
    }

    public override string Operation()
    {
        var results = new List<string>();
        foreach (var child in _children)
        {
            results.Add(child.Operation());
        }
        return $"Composite({string.Join("+", results)})";
    }
}

/*
// Usage
var leaf1 = new Leaf();
var leaf2 = new Leaf();
var composite1 = new Composite();
composite1.Add(leaf1);
composite1.Add(leaf2);
var composite2 = new Composite();
composite2.Add(composite1);
Console.WriteLine(composite2.Operation());  // Output: Composite(Leaf+Leaf)
*/
