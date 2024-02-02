import random
import numpy as np
from datetime import datetime, timedelta


class GenerateTestData:
    def __init__(self, start_date='2024-01-01', end_date='2024-12-31'):
        self.start_date = start_date
        self.end_date = end_date

    def generate_customer_data(self, num_customers=1000):
        def _generate_invoice_date():
            start = datetime.strptime(self.start_date, '%Y-%m-%d')
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
            random_date = start + \
                timedelta(days=random.randint(0, (end - start).days))
            return random_date.strftime('%Y-%m-%d')

        customer_data = []
        for _ in range(num_customers):
            customer_id = random.randint(1, 1000)
            invoice_date = _generate_invoice_date()
            count_of_items = random.randint(1, 10)
            revenue = round(random.uniform(10, 100), 2)
            customer_data.append(
                [customer_id, invoice_date, count_of_items, revenue])

        return np.array(customer_data)
