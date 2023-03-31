import polars as pl
import numpy as np
from typing import List, Dict


class MonteCarloSimulator:
    def __init__(self, data: pl.DataFrame):
        self.data = data

    def simulate(self, func: str, n_simulations: int, random_state: int = 42, **kwargs) -> List[float]:
        """Run a Monte Carlo simulation on the given function using the data provided.

        Args:
            func (str): The name of the function to simulate.
            n_simulations (int): The number of simulations to run.
            random_state (int): The random seed to use for the simulation.
            kwargs: Additional keyword arguments to pass to the function.

        Returns:
            List[float]: A list of the results from each simulation.
        """
        # Set the random seed
        np.random.seed(random_state)

        # Create an empty list to store the simulation results
        simulation_results = []

        # Loop through each simulation
        for i in range(n_simulations):
            # Sample data from the dataframe with replacement
            sampled_data = self.data.sample_n(len(self.data), replace=True)

            # Call the function with the sampled data and any additional keyword arguments
            result = getattr(sampled_data, func)(**kwargs)

            # Append the result to the simulation results
            simulation_results.append(result)

        return simulation_results

    def infer(self, simulation_results: List[float], alpha: float = 0.05) -> Dict[str, float]:
        """Perform statistical inference on the given simulation results.

        Args:
            simulation_results (List[float]): The results from the Monte Carlo simulation.
            alpha (float): The significance level to use for the hypothesis test.

        Returns:
            Dict[str, float]: A dictionary containing the inferential statistics.
        """
        # Calculate the mean and standard deviation of the simulation results
        mean = np.mean(simulation_results)
        std_dev = np.std(simulation_results)

        # Calculate the confidence interval
        lower_bound, upper_bound = np.percentile(
            simulation_results, [100 * alpha / 2, 100 * (1 - alpha / 2)])

        # Perform a hypothesis test
        null_hypothesis = 0  # Set the null hypothesis to zero
        p_value = 2 * min(np.mean(simulation_results <= null_hypothesis),
                          np.mean(simulation_results >= null_hypothesis))

        # Create a dictionary of the inferential statistics
        inferential_stats = {
            "mean": mean,
            "std_dev": std_dev,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "p_value": p_value
        }

        return inferential_stats

# Example usage


# Load new inference data into Polars DataFrame
data = pl.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [6, 7, 8, 9, 10],
    'y': [10, 20, 30, 40, 50]
})

# Instantiate MonteCarlo class
mc = MonteCarloSimulator(data=data, response='y', n_simulations=1000)

# Run Monte Carlo simulation
mc.run()

# Calculate mean and confidence intervals for each explanatory variable
means, ci_low, ci_high = mc.infer_means_ci()

# Print results
for col, mean, low, high in zip(data.columns[:-1], means, ci_low, ci_high):
    print(
        f"Variable: {col}, Mean: {mean:.2f}, 95% CI: ({low:.2f}, {high:.2f})")
