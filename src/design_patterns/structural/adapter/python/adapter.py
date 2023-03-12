class ExistingClass:
    def perform_computation(self, data: dict) -> float:
        # performs some computation using the data
        return result

class Adapter:
    def __init__(self, data: list):
        self.data = data

    def perform_computation(self) -> float:
        data_dict = self.convert_to_dict()
        existing = ExistingClass()
        return existing.perform_computation(data_dict)

    def convert_to_dict(self) -> dict:
        # converts the data from a list to a dictionary
        converted_data = {}
        return converted_data

# example usage
data = [1, 2, 3]
adapter = Adapter(data)
result = adapter.perform_computation()
