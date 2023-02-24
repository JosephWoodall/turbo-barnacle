interface IMediator
{
    void Send(string message, Colleague colleague);
}

class Mediator : IMediator
{
    private Colleague1 colleague1;
    private Colleague2 colleague2;

    public Mediator()
    {
        colleague1 = new Colleague1(this);
        colleague2 = new Colleague2(this);
    }

    public void Send(string message, Colleague colleague)
    {
        if (colleague == colleague1)
        {
            colleague2.Notify(message);
        }
        else
        {
            colleague1.Notify(message);
        }
    }
}

abstract class Colleague
{
    protected IMediator mediator;

    public Colleague(IMediator mediator)
    {
        this.mediator = mediator;
    }

    public abstract void Send(string message);

    public abstract void Notify(string message);
}

class Colleague1 : Colleague
{
    public Colleague1(IMediator mediator) : base(mediator) {}

    public override void Send(string message)
    {
        mediator.Send(message, this);
    }

    public override void Notify(string message)
    {
        Console.WriteLine($"Colleague1 received message: {message}");
    }
}

class Colleague2 : Colleague
{
    public Colleague2(IMediator mediator) : base(mediator) {}

    public override void Send(string message)
    {
        mediator.Send(message, this);
    }

    public override void Notify(string message)
    {
        Console.WriteLine($"Colleague2 received message: {message}");
    }
}

class Program
{
    static void Main(string[] args)
    {
        IMediator mediator = new Mediator();
        Colleague colleague1 = new Colleague1(mediator);
        Colleague colleague2 = new Colleague2(mediator);

        colleague1.Send("Hello from Colleague1!");
        colleague2.Send("Hi from Colleague2!");
    }
}
