interface IStrategy
{
    int DoOperation(int num1, int num2);
}

class OperationAdd : IStrategy
{
    public int DoOperation(int num1, int num2)
    {
        return num1 + num2;
    }
}

class OperationSubtract : IStrategy
{
    public int DoOperation(int num1, int num2)
    {
        return num1 - num2;
    }
}

class OperationMultiply : IStrategy
{
    public int DoOperation(int num1, int num2)
    {
        return num1 * num2;
    }
}

class Context
{
    private IStrategy strategy;

    public Context(IStrategy strategy)
    {
        this.strategy = strategy;
    }

    public int ExecuteStrategy(int num1, int num2)
    {
        return strategy.DoOperation(num1, num2);
    }
}
