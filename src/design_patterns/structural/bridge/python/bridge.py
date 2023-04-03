from abc import ABC, abstractmethod


# Implementor: defines the interface for implementation classes
class DrawingAPI(ABC):
    """ """
    @abstractmethod
    def draw_circle(self, x, y, radius):
        """

        :param x: param y:
        :param radius: param y:
        :param y: 

        """
        pass


# Concrete Implementor: provides concrete implementation for DrawingAPI interface
class DrawingAPI1(DrawingAPI):
    """ """
    def draw_circle(self, x, y, radius):
        """

        :param x: param y:
        :param radius: param y:
        :param y: 

        """
        print(f"API1.circle at {x}:{y} radius {radius}")


# Concrete Implementor: provides concrete implementation for DrawingAPI interface
class DrawingAPI2(DrawingAPI):
    """ """
    def draw_circle(self, x, y, radius):
        """

        :param x: param y:
        :param radius: param y:
        :param y: 

        """
        print(f"API2.circle at {x}:{y} radius {radius}")


# Abstraction: defines the abstraction's interface and maintains a reference to the implementor
class Shape(ABC):
    """ """
    def __init__(self, drawing_api):
        self.drawing_api = drawing_api

    @abstractmethod
    def draw(self):
        """ """
        pass

    @abstractmethod
    def resize(self, radius):
        """

        :param radius: 

        """
        pass


# Refined Abstraction: extends the abstraction interface to incorporate its own functionality
class CircleShape(Shape):
    """ """
    def __init__(self, x, y, radius, drawing_api):
        super().__init__(drawing_api)
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self):
        """ """
        self.drawing_api.draw_circle(self.x, self.y, self.radius)

    def resize(self, radius):
        """

        :param radius: 

        """
        self.radius = radius


# Client code
if __name__ == '__main__':
    # Create the concrete implementor instances
    api1 = DrawingAPI1()
    api2 = DrawingAPI2()

    # Create the abstraction instances with the implementor references
    circle1 = CircleShape(1, 2, 3, api1)
    circle2 = CircleShape(5, 7, 11, api2)

    # Draw the circles using their respective implementors
    circle1.draw()
    circle2.draw()
