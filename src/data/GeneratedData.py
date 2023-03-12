from faker import Faker
import uuid 
import random

class DataGenerator:
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.faker = Faker()
        self.data = {}
        
    def generate_data(self):
        for i in range(self.num_cols):
            self.data[f'col{i}'] = []
        for _ in range(self.num_rows):
            for i in range(self.num_cols):
                self.data[f'col{i}'].append(self.faker.random_element())
        return self.data
