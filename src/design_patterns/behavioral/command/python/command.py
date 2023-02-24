class Command:
    def execute(self):
        pass


class OpenFileCommand(Command):
    def __init__(self, filename):
        self.filename = filename

    def execute(self):
        print(f"Opening file {self.filename}")


class CloseFileCommand(Command):
    def __init__(self, filename):
        self.filename = filename

    def execute(self):
        print(f"Closing file {self.filename}")


class FileManager:
    def __init__(self):
        self.commands = []

    def add_command(self, command):
        self.commands.append(command)

    def execute_commands(self):
        for command in self.commands:
            command.execute()


file_manager = FileManager()
file_manager.add_command(OpenFileCommand("file1.txt"))
file_manager.add_command(CloseFileCommand("file1.txt"))
file_manager.add_command(OpenFileCommand("file2.txt"))
file_manager.execute_commands()
