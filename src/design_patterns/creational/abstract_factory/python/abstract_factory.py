from abc import ABC, abstractmethod

# Abstract Factory


class GUIFactory(ABC):
    """ """
    @abstractmethod
    def create_button(self):
        """ """
        pass

    @abstractmethod
    def create_checkbox(self):
        """ """
        pass

# Concrete Factory 1


class WinFactory(GUIFactory):
    """ """
    def create_button(self):
        """ """
        return WinButton()

    def create_checkbox(self):
        """ """
        return WinCheckbox()

# Concrete Factory 2


class MacFactory(GUIFactory):
    """ """
    def create_button(self):
        """ """
        return MacButton()

    def create_checkbox(self):
        """ """
        return MacCheckbox()

# Abstract Product A


class Button(ABC):
    """ """
    @abstractmethod
    def paint(self):
        """ """
        pass

# Concrete Product A1


class WinButton(Button):
    """ """
    def paint(self):
        """ """
        return "Windows button painted"

# Concrete Product A2


class MacButton(Button):
    """ """
    def paint(self):
        """ """
        return "Mac button painted"

# Abstract Product B


class Checkbox(ABC):
    """ """
    @abstractmethod
    def paint(self):
        """ """
        pass

# Concrete Product B1


class WinCheckbox(Checkbox):
    """ """
    def paint(self):
        """ """
        return "Windows checkbox painted"

# Concrete Product B2


class MacCheckbox(Checkbox):
    """ """
    def paint(self):
        """ """
        return "Mac checkbox painted"


# Client code
def client_code(factory: GUIFactory):
    """

    :param factory: GUIFactory: 

    """
    button = factory.create_button()
    checkbox = factory.create_checkbox()

    print(button.paint())
    print(checkbox.paint())


factory1 = WinFactory()
factory2 = MacFactory()

client_code(factory1)
client_code(factory2)
