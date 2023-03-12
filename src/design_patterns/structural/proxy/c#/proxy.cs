using System;

interface ISubject
{
    void Request();
}

class RealSubject : ISubject
{
    public void Request()
    {
        Console.WriteLine("RealSubject: Handling request.");
    }
}

class Proxy : ISubject
{
    private readonly RealSubject _realSubject;

    public Proxy(RealSubject realSubject)
    {
        _realSubject = realSubject;
    }

    public void Request()
    {
        if (CheckAccess())
        {
            _realSubject.Request();
            LogAccess();
        }
    }

    private bool CheckAccess()
    {
        Console.WriteLine("Proxy: Checking access before handling request.");
        return true;
    }

    private void LogAccess()
    {
        Console.WriteLine("Proxy: Logging the time of request.");
    }
}

/*
// Usage
RealSubject realSubject = new RealSubject();
Proxy proxy = new Proxy(realSubject);

proxy.Request();
*/