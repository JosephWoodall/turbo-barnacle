using System.Collections.Generic;

interface IObserver
{
    void Update(ISubject subject);
}

interface ISubject
{
    void Attach(IObserver observer);
    void Detach(IObserver observer);
    void Notify();
}

class ConcreteObserver : IObserver
{
    public void Update(ISubject subject)
    {
        ConcreteSubject concreteSubject = (ConcreteSubject)subject;
        System.Console.WriteLine($"Subject's state has changed to {concreteSubject.State}");
    }
}

class ConcreteSubject : ISubject
{
    private List<IObserver> observers = new List<IObserver>();
    private string state;

    public void Attach(IObserver observer)
    {
        observers.Add(observer);
    }

    public void Detach(IObserver observer)
    {
        observers.Remove(observer);
    }

    public void Notify()
    {
        foreach (IObserver observer in observers)
        {
            observer.Update(this);
        }
    }

    public string State
    {
        get { return state; }
        set
        {
            state = value;
            Notify();
        }
    }
}
