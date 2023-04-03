import copy


class Prototype:
    """ """
    def __init__(self):
        self.objects = {}

    def register(self, name, obj):
        """

        :param name: 
        :param obj: 

        """
        self.objects[name] = obj

    def unregister(self, name):
        """

        :param name: 

        """
        del self.objects[name]

    def clone(self, name, **attr):
        """

        :param name: 
        :param **attr: 

        """
        obj = copy.deepcopy(self.objects.get(name))
        obj.__dict__.update(attr)
        return obj


class Shape:
    """ """
    def __init__(self):
        self.id = None
        self.type = None

    def clone(self):
        """ """
        pass


class Rectangle(Shape):
    """ """
    def __init__(self):
        super().__init__()
        self.type = "rectangle"

    def clone(self):
        """ """
        return copy.deepcopy(self)


class Circle(Shape):
    """ """
    def __init__(self):
        super().__init__()
        self.type = "circle"

    def clone(self):
        """ """
        return copy.deepcopy(self)


prototype = Prototype()
prototype.register("rectangle", Rectangle())
prototype.register("circle", Circle())

rectangle = prototype.clone("rectangle")
circle = prototype.clone("circle")
print(rectangle.type)  # Output: rectangle
print(circle.type)  # Output: circle
