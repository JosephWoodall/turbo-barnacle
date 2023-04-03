class Strategy:
    """ """
    def do_operation(self, num1, num2):
        """

        :param num1: param num2:
        :param num2: 

        """
        pass


class OperationAdd(Strategy):
    """ """
    def do_operation(self, num1, num2):
        """

        :param num1: param num2:
        :param num2: 

        """
        return num1 + num2


class OperationSubtract(Strategy):
    """ """
    def do_operation(self, num1, num2):
        """

        :param num1: param num2:
        :param num2: 

        """
        return num1 - num2


class OperationMultiply(Strategy):
    """ """
    def do_operation(self, num1, num2):
        """

        :param num1: param num2:
        :param num2: 

        """
        return num1 * num2


class Context:
    """ """
    def __init__(self, strategy):
        self.strategy = strategy

    def execute_strategy(self, num1, num2):
        """

        :param num1: param num2:
        :param num2: 

        """
        return self.strategy.do_operation(num1, num2)
