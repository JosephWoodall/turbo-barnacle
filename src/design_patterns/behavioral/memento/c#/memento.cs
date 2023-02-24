class Memento
{
    private string state;

    public Memento(string state)
    {
        this.state = state;
    }

    public string GetState()
    {
        return state;
    }
}

class Originator
{
    private string state;

    public void SetState(string state)
    {
        this.state = state;
    }

    public Memento Save()
    {
        return new Memento(state);
    }

    public void Restore(Memento memento)
    {
        state = memento.GetState();
    }

    public string GetState()
    {
        return state;
    }
}

class Caretaker
{
    private Stack<Memento> mementos = new Stack<Memento>();
    private Originator originator;

    public Caretaker(Originator originator)
    {
        this.originator = originator;
    }

    public void SaveState()
    {
        mementos.Push(originator.Save());
    }

    public void RestoreState()
    {
        if (mementos.Count == 0)
            return;

        Memento memento = mementos.Pop();
        originator.Restore(memento);
    }
}
