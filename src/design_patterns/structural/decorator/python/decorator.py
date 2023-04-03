from abc import ABC, abstractmethod

# Component
class Component(ABC):
    """ """
    @abstractmethod
    def operation(self) -> str:
        """ """
        pass

# Concrete Component
class ConcreteComponent(Component):
    """ """
    def operation(self) -> str:
        """ """
        return "ConcreteComponent"

# Decorator
class Decorator(Component):
    """ """
    def __init__(self, component: Component) -> None:
        self._component = component

    @abstractmethod
    def operation(self) -> str:
        """ """
        pass

# Concrete Decorator A
class ConcreteDecoratorA(Decorator):
    """ """
    def operation(self) -> str:
        """ """
        return f"ConcreteDecoratorA({self._component.operation()})"

# Concrete Decorator B
class ConcreteDecoratorB(Decorator):
    """ """
    def operation(self) -> str:
        """ """
        return f"ConcreteDecoratorB({self._component.operation()})"

# Usage
component = ConcreteComponent()
decorator1 = ConcreteDecoratorA(component)
decorator2 = ConcreteDecoratorB(decorator1)
print(decorator2.operation())  # Output: ConcreteDecoratorB(ConcreteDecoratorA(ConcreteComponent))
