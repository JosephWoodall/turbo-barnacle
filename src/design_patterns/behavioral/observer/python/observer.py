class Observer:
    def update(self, subject):
        pass


class Subject:
    def __init__(self):
        self._observers = []

    def add_observer(self, observer):
        self._observers.append(observer)

    def remove_observer(self, observer):
        self._observers.remove(observer)

    def notify(self):
        for observer in self._observers:
            observer.update(self)


class ConcreteObserver(Observer):
    def update(self, subject):
        print(f"Subject's state has changed to {subject.state}")


class ConcreteSubject(Subject):
    def __init__(self):
        super().__init__()
        self.state = None

    def set_state(self, state):
        self.state = state
        self.notify()
