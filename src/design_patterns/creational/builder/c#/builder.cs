using System;

// "Product" class
class Pizza
{
    private string dough;
    private string sauce;
    private string topping;

    public void SetDough(string dough)
    {
        this.dough = dough;
    }

    public void SetSauce(string sauce)
    {
        this.sauce = sauce;
    }

    public void SetTopping(string topping)
    {
        this.topping = topping;
    }

    public void ShowPizza()
    {
        Console.WriteLine($"Pizza with {dough} dough, {sauce} sauce and {topping} topping");
    }
}

// "Builder" abstract class
abstract class PizzaBuilder
{
    protected Pizza pizza;

    public void CreateNewPizza()
    {
        pizza = new Pizza();
    }

    public Pizza GetPizza()
    {
        return pizza;
    }

    public abstract void BuildDough();
    public abstract void BuildSauce();
    public abstract void BuildTopping();
}

// "ConcreteBuilder" class
class MargheritaPizzaBuilder : PizzaBuilder
{
    public override void BuildDough()
    {
        pizza.SetDough("thin crust");
    }

    public override void BuildSauce()
    {
        pizza.SetSauce("tomato");
    }

    public override void BuildTopping()
    {
        pizza.SetTopping("mozzarella cheese");
    }
}

// "Director" class
class Cook
{
    private PizzaBuilder pizzaBuilder;

    public void SetPizzaBuilder(PizzaBuilder pizzaBuilder)
    {
        this.pizzaBuilder = pizzaBuilder;
    }

    public Pizza GetPizza()
    {
        return pizzaBuilder.GetPizza();
    }

    public void ConstructPizza()
    {
        pizzaBuilder.CreateNewPizza();
        pizzaBuilder.BuildDough();
        pizzaBuilder.BuildSauce();
        pizzaBuilder.BuildTopping();
    }
}

// Client code
class Client
{
    static void Main()
    {
        Cook cook = new Cook();
        PizzaBuilder builder = new MargheritaPizzaBuilder();

        cook.SetPizzaBuilder(builder);
        cook.ConstructPizza();

        Pizza pizza = cook.GetPizza();
        pizza.ShowPizza();
    }
}
