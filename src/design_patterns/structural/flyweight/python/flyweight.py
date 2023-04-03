from typing import Dict

class Flyweight:
    """ """
    def __init__(self, shared_state):
        self._shared_state = shared_state

    def operation(self, unique_state):
        """

        :param unique_state: 

        """
        return f"Flyweight: Shared '{self._shared_state}' and unique '{unique_state}'"

class FlyweightFactory:
    """ """
    _flyweights: Dict[str, Flyweight] = {}

    def __init__(self, initial_flyweights):
        for state in initial_flyweights:
            self._flyweights[self.get_key(state)] = Flyweight(state)

    def get_key(self, state):
        """

        :param state: 

        """
        return "_".join(sorted(state))

    def get_flyweight(self, shared_state):
        """

        :param shared_state: 

        """
        key = self.get_key(shared_state)
        if key not in self._flyweights:
            print("FlyweightFactory: Can't find a flyweight, creating new one.")
            self._flyweights[key] = Flyweight(shared_state)
        else:
            print("FlyweightFactory: Reusing existing flyweight.")
        return self._flyweights[key]

# Usage
flyweight_factory = FlyweightFactory([
    ["shared_state_1", "unique_state_1"],
    ["shared_state_2", "unique_state_2"],
    ["shared_state_3", "unique_state_3"]
])
flyweight1 = flyweight_factory.get_flyweight("shared_state_1")
flyweight1.operation("unique_state_1")  # Output: Flyweight: Shared 'shared_state_1' and unique 'unique_state_1'
flyweight2 = flyweight_factory.get_flyweight("shared_state_2")
flyweight2.operation("unique_state_2")  # Output: Flyweight: Shared 'shared_state_2' and unique 'unique_state_2'
flyweight3 = flyweight_factory.get_flyweight("shared_state_1")
flyweight3.operation("unique_state_3")  # Output: Flyweight: Shared 'shared_state_1' and unique 'unique_state_3'
