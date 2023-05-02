from kfp import dsl


class Cleaning:

    def __init__(self, *arg):
        print("-----CLEANING INITIALIZED-----")
        self.arg = arg

    @dsl.component
    def _cleaning_process_one(self, a: int) -> int:
        success = a
        return (success * 2)
