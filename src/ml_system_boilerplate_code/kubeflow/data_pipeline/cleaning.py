from kfp import dsl


class Cleaning:

    def __init__(self, *arg):
        print("-----CLEANING INITIALIZED-----")
        self.arg = arg
    
    @dsl.component
    def _cleaning_process_one(self) -> str:
        print("cleaning process one has run")


if __name__ == '__main__':
    Cleaning()
