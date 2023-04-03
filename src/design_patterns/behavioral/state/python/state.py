class State:
    """ """
    def handle(self, context):
        """

        :param context: 

        """
        pass


class ConcreteStateA(State):
    """ """
    def handle(self, context):
        """

        :param context: 

        """
        print("Handling request with ConcreteStateA")
        context.state = ConcreteStateB()


class ConcreteStateB(State):
    """ """
    def handle(self, context):
        """

        :param context: 

        """
        print("Handling request with ConcreteStateB")
        context.state = ConcreteStateA()


class Context:
    """ """
    def __init__(self):
        self.state = ConcreteStateA()

    def request(self):
        """ """
        self.state.handle(self)
