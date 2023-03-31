import polars as pl
import numpy as np


class MonteCarloSimulator:
    """
    A Monte Carlo simulator for inferential statistics.
    """

    def __init__(self, sample_size, null_mean, null_std, alternative_mean, alternative_std, alpha, num_simulations):
        """
        Initializes the MonteCarloSimulator object.

        Args:
            sample_size (int): The sample size.
            null_mean (float): The null hypothesis mean.
            null_std (float): The null hypothesis standard deviation.
            alternative_mean (float): The alternative hypothesis mean.
            alternative_std (float): The alternative hypothesis standard deviation.
            alpha (float): The significance level.
            num_simulations (int): The number of simulations to run.
        """
        self.sample_size = sample_size
        self.null_mean = null_mean
        self.null_std = null_std
        self.alternative_mean = alternative_mean
        self.alternative_std = alternative_std
        self.alpha = alpha
        self.num_simulations = num_simulations

    def run_simulation(self):
        """
        Runs the Monte Carlo simulation and returns the p-value.
        """
        # Generate a normal distribution with the null hypothesis parameters
        null_distribution = pl.Series(np.random.normal(
            self.null_mean, self.null_std, self.sample_size * self.num_simulations))

        # Generate a normal distribution with the alternative hypothesis parameters
        alternative_distribution = pl.Series(np.random.normal(
            self.alternative_mean, self.alternative_std, self.sample_size * self.num_simulations))

        # Calculate the t-statistic for each sample
        null_sample_means = null_distribution.reshape(
            self.num_simulations, self.sample_size).mean(axis=1)
        alternative_sample_means = alternative_distribution.reshape(
            self.num_simulations, self.sample_size).mean(axis=1)
        t_values = (alternative_sample_means - self.null_mean) / \
            (self.null_std / np.sqrt(self.sample_size))

        # Calculate the p-value
        if self.alternative_mean > self.null_mean:
            p_value = (t_values > pl.Series(self.alpha)
                       ).sum() / self.num_simulations
        else:
            p_value = (t_values < pl.Series(-self.alpha)
                       ).sum() / self.num_simulations

        return p_value


# Initialize the MonteCarloSimulator object
simulator = MonteCarloSimulator(
    sample_size=10,
    null_mean=100,
    null_std=10,
    alternative_mean=105,
    alternative_std=10,
    alpha=0.05,
    num_simulations=1000
)

# Run the simulation
p_value = simulator.run_simulation()

# Print the result
print(f"p-value: {p_value}")
