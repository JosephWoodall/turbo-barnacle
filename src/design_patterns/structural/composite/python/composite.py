from abc import ABC, abstractmethod

# Component
class Component(ABC):
    """ """
    @abstractmethod
    def operation(self) -> str:
        """ """
        pass

# Leaf
class Leaf(Component):
    """ """
    def operation(self) -> str:
        """ """
        return "Leaf"

# Composite
class Composite(Component):
    """ """
    def __init__(self) -> None:
        self._children = []

    def add(self, component: Component) -> None:
        """

        :param component: Component:
        :param component: Component:
        :param component: Component:
        :param component: Component: 

        """
        self._children.append(component)

    def remove(self, component: Component) -> None:
        """

        :param component: Component:
        :param component: Component:
        :param component: Component:
        :param component: Component: 

        """
        self._children.remove(component)

    def operation(self) -> str:
        """ """
        results = []
        for child in self._children:
            results.append(child.operation())
        return f"Composite({'+'.join(results)})"

# Usage
leaf1 = Leaf()
leaf2 = Leaf()
composite1 = Composite()
composite1.add(leaf1)
composite1.add(leaf2)
composite2 = Composite()
composite2.add(composite1)
print(composite2.operation())  # Output: Composite(Leaf+Leaf)
