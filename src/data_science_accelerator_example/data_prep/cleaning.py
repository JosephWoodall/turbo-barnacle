class Cleaning:
    def __init__(self, data):
        self.data = data

    def remove_duplicates(self):
        self.data.drop_duplicates(inplace=True)

    def remove_null_values(self):
        self.data.dropna(inplace=True)

    def fill_null_values(self, value):
        self.data.fillna(value, inplace=True)
