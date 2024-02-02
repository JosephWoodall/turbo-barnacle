#include <iostream>
#include <vector>
#include <string>

using namespace std;

struct Todo
{
    string description;
    bool completed;
};

void add_todo(vector<Todo> &todos)
{
    string description;
    cout << "Enter todo description: ";
    getline(cin, description);
    todos.push_back({description, false});
    cout << "Todo added!" << endl;
}

void delete_todo(vector<Todo> &todos)
{
    int todo_number;
    cout << "Enter the number of the todo to delete: ";
    cin >> todo_number;

    if (todo_number <= 0 || todo_number > todos.size())
    {
        cout << "Invalid todo number." << endl;
        return;
    }

    todos.erase(todos.begin() + todo_number - 1);
    cout << "Todo deleted." << endl;
}

void list_todos(const vector<Todo> &todos)
{
    if (todos.empty())
    {
        cout << "No todos found." << endl;
        return;
    }

    cout << "Todos:" << endl;
    for (int i = 0; i < todos.size(); ++i)
    {
        cout << i + 1 << ". " << todos[i].description << " (";
        if (todos[i].completed)
        {
            cout << "completed";
        }
        else
        {
            cout << "pending";
        }
        cout << ")" << endl;
    }
}

void mark_completed(vector<Todo> &todos)
{
    int todo_number;
    cout << "Enter the number of the todo to mark as completed: ";
    cin >> todo_number;

    if (todo_number <= 0 || todo_number > todos.size())
    {
        cout << "Invalid todo number." << endl;
        return;
    }

    todos[todo_number - 1].completed = true;
    cout << "Todo marked as completed." << endl;
}

int main()
{
    vector<Todo> todos;

    int choice;
    do
    {
        cout << "\nMenu:" << endl;
        cout << "1. Add todo" << endl;
        cout << "2. List todos" << endl;
        cout << "3. Mark todo as completed" << endl;
        cout << "4. Delete todo" << endl;
        cout << "5. Exit" << endl;
        cout << "Enter your choice: ";
        cin >> choice;

        switch (choice)
        {
        case 1:
            add_todo(todos);
            break;
        case 2:
            list_todos(todos);
            break;
        case 3:
            mark_completed(todos);
            break;
        case 4:
            delete_todo(todos);
            break;
        case 5:
            cout << "Exiting..." << endl;
            break;
        default:
            cout << "Invalid choice." << endl;
        }
    } while (choice != 5);

    return 0;
}
