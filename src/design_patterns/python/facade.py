'''
Facade is a structural pattern, and is the interface that programmers interact with, and hides the complexity from the end users
    - it's usually a wrapper class that abstracts lower-level details 
    - examples include: 
        - http clients
        - dynamic arrays like Vectors or ArrayLists
'''


class Array:
    def __init__(self):
        self.capacity = 2
        self.length = 0
        self.arr = [0] * 2  # Array of capacity = 2

    def pushback(self, n):
        if self.length == self.capacity:
            self.resize()
        # insert at next empty position
        self.arr[self.length] = n
        self.length += 1
