import great_expectations as ge

class DataValidation:
    def __init__(self, config_path):
        self.suite = ge.from_config(config_path)
        
    def run_validations(self, data):
        validation_result = self.suite.validate(data)
        if validation_result.success:
            print("Data validation succeeded.")
        else:
            print("Data validation failed:")
            print(validation_result.exception_info)