import inspect


class Cleaning:

    def __init__(self, *arg):
        print("-----CLEANING INITIALIZED-----")
        self.arg = arg

    def _hashed_features(self):
        """
        _hashed_feature is a bucket of deterministic and portable hash of string representation and accept the trade-off of collisions in the data representation.

        This function solves problems associated with categorical features such as incomplete vocabulary, model size due to cardinality, and cold start.

        """
        hased_features = hash(frozenset(self.arg.items()))
        return hased_features

    def _embedding(self):
        """
        _embedding learns a data representation that maps high-cardinality data into a lower dimensional space in such a way that the information relevant to the learning problem is solved.

        This function solves the problem of high-cardinality features where closeness relationships are important to preserve.
        """


if __name__ == '__main__':
    Cleaning()
