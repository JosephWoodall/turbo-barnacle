#include <iostream>
#include <vector>
#include <string>

using namespace std;

// Structure that will represent a contact/entry in the file management system
struct Contact
{
    string name;
    string phoneNumber;
    string email;
};

// Function that will add a new contact to the file management system
void add_contact(vector<Contact> &contacts)
{
    Contact new_contact;
    cout << "Enter the name of the contact:" << endl;
    getline(cin, new_contact.name);
    cout << "Enter the phone number of the contact:";
    getline(cin, new_contact.phoneNumber);
    cout << "\n";
    cout << "Enter the email of the contact: ";
    contacts.push_back(new_contact);
    cout << "Contact added successfully!" << endl;
}

// Function that will list all of the contacts in the file management system
void list_contacts(const vector<Contact> &contacts)
{
    if (contacts.empty())
    {
        cout << "No contacts found." << endl;
        return;
    }

    cout << "Contacts:" << endl;
    for (const Contact &contact : contacts)
    {
        cout << "Name: " << contact.name << endl;
        cout << "Phone Number: " << contact.phoneNumber << endl;
        cout << "Email: " << contact.email << endl;
        cout << endl;
    }
}

int main()
{
    vector<Contact> contacts;

    int choice;
    do
    {
        cout << "\nMenu:" << endl;
        cout << "1. Add contact" << endl;
        cout << "2. List contacts" << endl;
        cout << "3. Exit" << endl;
        cout << "Enter your choice: ";
        cin >> choice;

        switch (choice)
        {
        case 1:
            add_contact(contacts);
            break;
        case 2:
            list_contacts(contacts);
            break;
        case 3:
            cout << "Exiting..." << endl;
            break;
        default:
            cout << "Invalid choice." << endl;
        }
    } while (choice != 3);

    return 0;
}