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

"""
# The below code is deprecated, but I wanted to keep it for records anyways
class GeneratedData:
    
    def __init__(self):
        self.data = {}
        
    def buildDictionary(self, length):
        for i in range(length):
            self.user = {uuid.uuid4(): {
                'First_Name' : Faker().first_name()
                ,'Last_Name' : Faker().last_name()
                , 'profession' : Faker().job()
                , 'company' : Faker().company()
                ,'aba' : Faker().aba()
                ,'bban' : Faker().bban()
                ,'iban' : Faker().iban()
                , 'credit_card_number' : Faker().credit_card_number()
                , 'credit_card_provider' : Faker().credit_card_provider()
                , 'transaction_latitude' : Faker().latitude()
                , 'transaction_longitude' : Faker().longitude()
                , 'transaction_medium' : Faker().user_agent()
                , 'password' : Faker().pystr()
                , 'has_debt' : Faker().pybool()
                , 'last_payment_amount_USD' : Faker().pyint()
                ,'x1' : Faker().pyint()
                ,'x2' : Faker().pyint()
                ,'x3' : Faker().pyint()
                ,'x4' : Faker().pyint()
                ,'x5' : Faker().pyint()
                ,'x6' : Faker().pyint()
                ,'x7' : Faker().pyint()
                ,'x8' : Faker().pyint()
                ,'x9' : Faker().pyint()
                ,'x10' : Faker().pyint()
                ,'x11' : Faker().pyint()
                ,'x12' : Faker().pyint()
                ,'x3' : Faker().pyint()
                ,'x14' : Faker().pyint()
                ,'x15' : Faker().pyint()
                ,'x16' : Faker().pyint()
                ,'x17' : Faker().pyint()
                ,'x18' : Faker().pyint()
                ,'x19' : Faker().pyint()
                ,'x20' : Faker().pyint()
            }}  
            self.data.update(self.user)
        return self.data
        #print(self.data)
"""
