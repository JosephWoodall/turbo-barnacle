'''
Strategy is a behavioral pattern, and lets you modify or extend the behavior of a class
    - a class is open (for extension) and closed (for modification)
'''

from abc import ABC, abstractmethod


class FilterStrategy(ABC):

    @abstractmethod
    def removeValue(self, val):
        pass


class RemoveNegativeStrategy(FilterStrategy):
    def removeValue(self, val):
        return val < 0


class RemoveOddStrategy(FilterStrategy):
    def removeValue(self, val):
        return abs(val) % 2


class Values:
    def __init__(self, vals):
        self.vals = vals

    def filter(self, strategy):
        res = []
        for n in self.values:
            if not strategy.removeValue(n):
                res.append(n)
        return res


values = Values([-7, -4, -1, 0, 2, 6, 9])
print(values.filter(RemoveNegativeStrategy))  # [0, 2, 6, 9]
print(values.filter(RemoveOddStrategy))  # [-4, 0, 2, 6]
