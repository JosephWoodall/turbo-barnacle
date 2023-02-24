abstract class Visitor
{
    public virtual void VisitElement(Element element) { }

    public virtual void VisitConcreteElementA(ConcreteElementA elementA) { }

    public virtual void VisitConcreteElementB(ConcreteElementB elementB) { }
}

abstract class Element
{
    public abstract void Accept(Visitor visitor);
}

class ConcreteElementA : Element
{
    public override void Accept(Visitor visitor)
    {
        visitor.VisitConcreteElementA(this);
    }
}

class ConcreteElementB : Element
{
    public override void Accept(Visitor visitor)
    {
        visitor.VisitConcreteElementB(this);
    }
}

class ConcreteVisitorA : Visitor
{
    public override void VisitConcreteElementA(ConcreteElementA elementA)
    {
        Console.WriteLine("ConcreteVisitorA visited ConcreteElementA");
    }

    public override void VisitElement(Element element)
    {
        Console.WriteLine("ConcreteVisitorA visited Element");
    }
}

class ConcreteVisitorB : Visitor
{
    public override void VisitConcreteElementB(ConcreteElementB elementB)
    {
        Console.WriteLine("ConcreteVisitorB visited ConcreteElementB");
    }

    public override void VisitElement(Element element)
    {
        Console.WriteLine("ConcreteVisitorB visited Element");
    }
}
