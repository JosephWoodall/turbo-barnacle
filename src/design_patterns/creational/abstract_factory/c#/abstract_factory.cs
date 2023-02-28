using System;

// Abstract Factory interface
public interface IAbstractFactory
{
    IProductA CreateProductA();
    IProductB CreateProductB();
}

// Concrete Factory 1
public class ConcreteFactory1 : IAbstractFactory
{
    public IProductA CreateProductA()
    {
        return new ProductA1();
    }

    public IProductB CreateProductB()
    {
        return new ProductB1();
    }
}

// Concrete Factory 2
public class ConcreteFactory2 : IAbstractFactory
{
    public IProductA CreateProductA()
    {
        return new ProductA2();
    }

    public IProductB CreateProductB()
    {
        return new ProductB2();
    }
}

// Abstract Product A interface
public interface IProductA
{
    string GetName();
}

// Concrete Product A1
public class ProductA1 : IProductA
{
    public string GetName()
    {
        return "Product A1";
    }
}

// Concrete Product A2
public class ProductA2 : IProductA
{
    public string GetName()
    {
        return "Product A2";
    }
}

// Abstract Product B interface
public interface IProductB
{
    string GetName();
}

// Concrete Product B1
public class ProductB1 : IProductB
{
    public string GetName()
    {
        return "Product B1";
    }
}

// Concrete Product B2
public class ProductB2 : IProductB
{
    public string GetName()
    {
        return "Product B2";
    }
}

// Client class
public class Client
{
    private readonly IAbstractFactory factory;

    public Client(IAbstractFactory factory)
    {
        this.factory = factory;
    }

    public void Run()
    {
        IProductA productA = factory.CreateProductA();
        IProductB productB = factory.CreateProductB();

        Console.WriteLine(productA.GetName());
        Console.WriteLine(productB.GetName());
    }
}

// Usage
public class Program
{
    public static void Main()
    {
        // Client uses ConcreteFactory1
        Client client1 = new Client(new ConcreteFactory1());
        client1.Run();

        // Client uses ConcreteFactory2
        Client client2 = new Client(new ConcreteFactory2());
        client2.Run();
    }
}
