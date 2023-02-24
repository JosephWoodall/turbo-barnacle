interface IIterator
{
    bool HasNext();
    object Next();
}

class Iterator : IIterator
{
    private object[] collection;
    private int index;

    public Iterator(object[] collection)
    {
        this.collection = collection;
        this.index = 0;
    }

    public bool HasNext()
    {
        return index < collection.Length;
    }

    public object Next()
    {
        object item = collection[index];
        index++;
        return item;
    }
}

interface IAggregate
{
    IIterator CreateIterator();
}

class Collection : IAggregate
{
    private object[] items;

    public Collection()
    {
        items = new object[0];
    }

    public void AddItem(object item)
    {
        Array.Resize(ref items, items.Length + 1);
        items[items.Length - 1] = item;
    }

    public IIterator CreateIterator()
    {
        return new Iterator(items);
    }
}

Collection collection = new Collection();
collection.AddItem("Item 1");
collection.AddItem("Item 2");
collection.AddItem("Item 3");

IIterator iterator = collection.CreateIterator();
while (iterator.HasNext())
{
    object item = iterator.Next();
    Console.WriteLine(item);
}
