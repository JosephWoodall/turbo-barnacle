class Command {
    execute() { }
}

class OpenFileCommand extends Command {
    constructor(filename) {
        super();
        this.filename = filename;
    }

    execute() {
        console.log(`Opening file ${this.filename}`);
    }
}

class CloseFileCommand extends Command {
    constructor(filename) {
        super();
        this.filename = filename;
    }

    execute() {
        console.log(`Closing file ${this.filename}`);
    }
}

class FileManager {
    constructor() {
        this.commands = [];
    }

    addCommand(command) {
        this.commands.push(command);
    }

    executeCommands() {
        for (let command of this.commands) {
            command.execute();
        }
    }
}

let fileManager = new FileManager();
fileManager.addCommand(new OpenFileCommand("file1.txt"));
fileManager.addCommand(new CloseFileCommand("file1.txt"));
fileManager.addCommand(new OpenFileCommand("file2.txt"));
fileManager.executeCommands();