abstract class Handler
{
    protected Handler successor;

    public void SetSuccessor(Handler successor)
    {
        this.successor = successor;
    }

    public abstract void HandleRequest(string request);
}

class ConcreteHandler1 : Handler
{
    public override void HandleRequest(string request)
    {
        if (request == "request 1")
        {
            Console.WriteLine("Handled by ConcreteHandler1");
        }
        else if (successor != null)
        {
            successor.HandleRequest(request);
        }
    }
}

class ConcreteHandler2 : Handler
{
    public override void HandleRequest(string request)
    {
        if (request == "request 2")
        {
            Console.WriteLine("Handled by ConcreteHandler2");
        }
        else if (successor != null)
        {
            successor.HandleRequest(request);
        }
    }
}

class ConcreteHandler3 : Handler
{
    public override void HandleRequest(string request)
    {
        if (request == "request 3")
        {
            Console.WriteLine("Handled by ConcreteHandler3");
        }
        else if (successor != null)
        {
            successor.HandleRequest(request);
        }
    }
}

// Client code
Handler handler1 = new ConcreteHandler1();
Handler handler2 = new ConcreteHandler2();
Handler handler3 = new ConcreteHandler3();
handler1.SetSuccessor(handler2);
handler2.SetSuccessor(handler3);

handler1.HandleRequest("request 1");
handler1.HandleRequest("request 2");
handler1.HandleRequest("request 3");
handler1.HandleRequest("unknown request");