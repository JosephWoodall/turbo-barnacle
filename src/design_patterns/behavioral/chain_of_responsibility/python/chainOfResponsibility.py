class Handler:
    """ """
    def __init__(self, successor=None):
        self._successor = successor

    def handle_request(self, request):
        """

        :param request: 

        """
        if self._successor is not None:
            return self._successor.handle_request(request)


class ConcreteHandler1(Handler):
    """ """
    def handle_request(self, request):
        """

        :param request: 

        """
        if request == 'request 1':
            print('Handled by ConcreteHandler1')
        else:
            super().handle_request(request)


class ConcreteHandler2(Handler):
    """ """
    def handle_request(self, request):
        """

        :param request: 

        """
        if request == 'request 2':
            print('Handled by ConcreteHandler2')
        else:
            super().handle_request(request)


class ConcreteHandler3(Handler):
    """ """
    def handle_request(self, request):
        """

        :param request: 

        """
        if request == 'request 3':
            print('Handled by ConcreteHandler3')
        else:
            super().handle_request(request)


# Client code
handler1 = ConcreteHandler1(ConcreteHandler2(ConcreteHandler3()))
handler1.handle_request('request 1')
handler1.handle_request('request 2')
handler1.handle_request('request 3')
handler1.handle_request('unknown request')
