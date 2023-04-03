class Iterator:
    """ """
    def __init__(self, collection):
        self.collection = collection
        self.index = 0

    def has_next(self):
        """ """
        return self.index < len(self.collection)

    def next(self):
        """ """
        item = self.collection[self.index]
        self.index += 1
        return item


class Collection:
    """ """
    def __init__(self):
        self.items = []

    def add_item(self, item):
        """

        :param item: 

        """
        self.items.append(item)

    def iterator(self):
        """ """
        return Iterator(self.items)


collection = Collection()
collection.add_item("Item 1")
collection.add_item("Item 2")
collection.add_item("Item 3")

iterator = collection.iterator()
while iterator.has_next():
    item = iterator.next()
    print(item)
