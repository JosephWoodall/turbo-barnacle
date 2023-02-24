abstract class AbstractClass
{
    public void TemplateMethod()
    {
        Operation1();
        Operation2();
        Operation3();
    }

    protected virtual void Operation1() { }

    protected virtual void Operation2() { }

    protected virtual void Operation3() { }
}

class ConcreteClass : AbstractClass
{
    protected override void Operation1()
    {
        Console.WriteLine("ConcreteClass.Operation1() called");
    }

    protected override void Operation2()
    {
        Console.WriteLine("ConcreteClass.Operation2() called");
    }
}

class AnotherConcreteClass : AbstractClass
{
    protected override void Operation1()
    {
        Console.WriteLine("AnotherConcreteClass.Operation1() called");
    }

    protected override void Operation3()
    {
        Console.WriteLine("AnotherConcreteClass.Operation3() called");
    }
}
