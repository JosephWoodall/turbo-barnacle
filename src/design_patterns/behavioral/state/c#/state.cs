interface IState
{
    void Handle(Context context);
}

class ConcreteStateA : IState
{
    public void Handle(Context context)
    {
        System.Console.WriteLine("Handling request with ConcreteStateA");
        context.State = new ConcreteStateB();
    }
}

class ConcreteStateB : IState
{
    public void Handle(Context context)
    {
        System.Console.WriteLine("Handling request with ConcreteStateB");
        context.State = new ConcreteStateA
