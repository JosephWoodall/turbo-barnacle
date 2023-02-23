'''
Iterator is a behavioral pattern, that defines how the values in an object can be iterated through
'''

# Python has a list iterator built in
myList = [1, 2, 3, 4]
for n in myList:
    print(n)

# But, here is a custom iterator for iterating through some other complex objects like binary trees and linked lists


class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


class LinkedList:
    def __init__(self, head):
        self.head = head
        self.cur = None

    # Defines the iterator
    def __iter__(self):
        self.cur = self.head
        return self

    # Iterate
    def __next__(self):
        if self.cur:
            val = self.cur.val
            self.cur = self.cur.next
            return val
        else:
            raise StopIteration


# Initialize a linked list
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
myList = LinkedList(head)

# Iterate through linked list
for n in myList:
    print(n)
