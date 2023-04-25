import inspect


class Cleaning:

    def __init__(self, *arg):
        print("-----CLEANING INITIALIZED-----")
        self.arg = arg

    def _hashed_features(self):
        """
        _hashed_feature solves problems associated with categorical features such as incomplete vocabulary, model size due to cardinality, and cold start.

        The solution is to bucket a deterministic and portable hash of string representation and accept the trade-off of collisions in the data representation.
        """
        hased_features = hash(frozenset(self.arg.items()))
        return hased_features


if __name__ == '__main__':
    Cleaning()
