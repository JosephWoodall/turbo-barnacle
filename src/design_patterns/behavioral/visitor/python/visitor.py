class Visitor:
    """ """
    def visit_element(self, element):
        """

        :param element: 

        """
        pass


class Element:
    """ """
    def accept(self, visitor):
        """

        :param visitor: 

        """
        visitor.visit_element(self)


class ConcreteElementA(Element):
    """ """
    def accept(self, visitor):
        """

        :param visitor: 

        """
        visitor.visit_concrete_element_a(self)


class ConcreteElementB(Element):
    """ """
    def accept(self, visitor):
        """

        :param visitor: 

        """
        visitor.visit_concrete_element_b(self)


class ConcreteVisitorA(Visitor):
    """ """
    def visit_concrete_element_a(self, element):
        """

        :param element: 

        """
        print("ConcreteVisitorA visited ConcreteElementA")

    def visit_element(self, element):
        """

        :param element: 

        """
        print("ConcreteVisitorA visited Element")


class ConcreteVisitorB(Visitor):
    """ """
    def visit_concrete_element_b(self, element):
        """

        :param element: 

        """
        print("ConcreteVisitorB visited ConcreteElementB")

    def visit_element(self, element):
        """

        :param element: 

        """
        print("ConcreteVisitorB visited Element")
