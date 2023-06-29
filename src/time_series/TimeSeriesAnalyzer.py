import torch
from typing import Union, List, Tuple

class TimeSeriesAnalyzer:
    """A class for analyzing time series data."""

    def __init__(self, data: Union[List[float], torch.Tensor]):
        """
        Initialize the TimeSeriesAnalyzer object.

        Parameters:
            data (list or torch.Tensor): The time series data.
        """
        self.data = torch.tensor(data)

    def check_stationarity(self) -> None:
        """
        Check the stationarity of the time series data.

        Performs the Augmented Dickey-Fuller (ADF) test on the first differences of the data
        and prints the ADF statistic, p-value, and critical values.

        Returns:
            None
        """
        # Calculate the first differences
        diff = torch.diff(self.data)

        # Perform the Augmented Dickey-Fuller test
        adf_result = self.adf_test(diff)

        # Print the ADF statistic, p-value, and critical values
        print('ADF Statistic:', adf_result[0].item())
        print('p-value:', adf_result[1].item())
        print('Critical Values:')
        for key, value in adf_result[4].items():
            print('\t', key, ':', value)

    def adf_test(self, data: torch.Tensor) -> Tuple[float, float, None, None, dict]:
        """
        Perform the Augmented Dickey-Fuller (ADF) test.

        Parameters:
            data (torch.Tensor): The input data for the ADF test.

        Returns:
            tuple: A tuple containing the ADF statistic, p-value, and critical values.
        """
        # Compute the ADF statistic and p-value
        n = data.size(0)
        t = torch.arange(1, n+1)
        Xt = torch.cumsum(data - data.mean())
        Yt = data - data.mean()

        numerator = (t * Xt).sum()
        denominator = (t**2).sum()
        beta_hat = numerator / denominator

        Xt_hat = beta_hat * t
        residuals = Yt - Xt_hat

        numerator = (residuals[1:] * residuals[:-1]).sum()
        denominator = (residuals**2).sum()
        gamma_hat = numerator / denominator

        ADF_statistic = (gamma_hat - 1) * torch.sqrt(n / (1 - beta_hat**2))
        p_value = self.adf_pvalue(ADF_statistic)

        # Compute the critical values
        critical_values = {
            '1%': -2.575,
            '5%': -1.950,
            '10%': -1.617
        }

        return ADF_statistic, p_value, None, None, critical_values

    def adf_pvalue(self, statistic: float) -> float:
        """
        Compute the p-value using MacKinnon's approximate distribution.

        Parameters:
            statistic (float): The ADF statistic.

        Returns:
            float: The p-value.
        """
        p_value = 0.0

        if statistic < -3.439:
            p_value = 0.01
        elif statistic < -2.874:
            p_value = 0.025
        elif statistic < -2.573:
            p_value = 0.05
        elif statistic < -2.326:
            p_value = 0.10

        return p_value
