abstract class Command
{
    public abstract void Execute();
}

class OpenFileCommand : Command
{
    private string filename;

    public OpenFileCommand(string filename)
    {
        this.filename = filename;
    }

    public override void Execute()
    {
        Console.WriteLine($"Opening file {filename}");
    }
}

class CloseFileCommand : Command
{
    private string filename;

    public CloseFileCommand(string filename)
    {
        this.filename = filename;
    }

    public override void Execute()
    {
        Console.WriteLine($"Closing file {filename}");
    }
}

class FileManager
{
    private List<Command> commands = new List<Command>();

    public void AddCommand(Command command)
    {
        commands.Add(command);
    }

    public void ExecuteCommands()
    {
        foreach (Command command in commands)
        {
            command.Execute();
        }
    }
}

FileManager fileManager = new FileManager();
fileManager.AddCommand(new OpenFileCommand("file1.txt"));
fileManager.AddCommand(new CloseFileCommand("file1.txt"));
fileManager.AddCommand(new OpenFileCommand("file2.txt"));
fileManager.ExecuteCommands();