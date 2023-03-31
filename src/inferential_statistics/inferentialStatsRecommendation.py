class InferentialStatsRecommendation:
    def __init__(self):
        pass

    def inferential_test_suggestion(self):
        # Prompt the user for information about their data
        data_type = input(
            "What type of data are you working with? (continuous/categorical) ")
        if data_type == "continuous":
            n_samples = int(input("How many samples do you have? "))
            if n_samples == 1:
                return "one-sample t-test"
            elif n_samples == 2:
                independent = input("Are the samples independent or paired? ")
                if independent == "independent":
                    return "two-sample t-test"
                elif independent == "paired":
                    return "paired t-test"
                else:
                    return "Invalid input. Please enter 'independent' or 'paired'."
            elif n_samples > 2:
                return "ANOVA"
            else:
                return "Invalid input. Please enter a positive integer for the number of samples."
        elif data_type == "categorical":
            n_variables = int(input("How many variables do you have? "))
            if n_variables == 1:
                return "Chi-squared goodness-of-fit test"
            elif n_variables == 2:
                return "Chi-squared test of independence"
            elif n_variables > 2:
                return "Logistic regression"
            else:
                return "Invalid input. Please enter a positive integer for the number of variables."
        else:
            return "Invalid input. Please enter 'continuous' or 'categorical'."


# Example usage
test_suggester = InferentialStatsRecommendation()
test_suggestion = test_suggester.inferential_test_suggestion()
print("Suggested test:", test_suggestion)
