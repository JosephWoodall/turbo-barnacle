class Memento:
    def __init__(self, state):
        self._state = state

    def get_state(self):
        return self._state


class Originator:
    def __init__(self):
        self._state = None

    def set_state(self, state):
        self._state = state

    def save(self):
        return Memento(self._state)

    def restore(self, memento):
        self._state = memento.get_state()


class Caretaker:
    def __init__(self, originator):
        self._mementos = []
        self._originator = originator

    def save_state(self):
        self._mementos.append(self._originator.save())

    def restore_state(self):
        if not self._mementos:
            return

        memento = self._mementos.pop()
        self._originator.restore(memento)
