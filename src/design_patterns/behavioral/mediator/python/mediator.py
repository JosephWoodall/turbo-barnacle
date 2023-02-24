class Mediator:
    def __init__(self):
        self.colleague1 = Colleague1(self)
        self.colleague2 = Colleague2(self)

    def send(self, message, colleague):
        if colleague == self.colleague1:
            self.colleague2.notify(message)
        else:
            self.colleague1.notify(message)


class Colleague1:
    def __init__(self, mediator):
        self.mediator = mediator

    def send(self, message):
        self.mediator.send(message, self)

    def notify(self, message):
        print(f"Colleague1 received message: {message}")


class Colleague2:
    def __init__(self, mediator):
        self.mediator = mediator

    def send(self, message):
        self.mediator.send(message, self)

    def notify(self, message):
        print(f"Colleague2 received message: {message}")


mediator = Mediator()
colleague1 = Colleague1(mediator)
colleague2 = Colleague2(mediator)

colleague1.send("Hello from Colleague1!")
colleague2.send("Hi from Colleague2!")
