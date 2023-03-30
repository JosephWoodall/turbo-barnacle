import polars as pl
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots


class Dashboard:
    def __init__(self, data):
        self.data = data

    def generate_dashboard(self):
        # Get some basic statistics about the data
        summary = self.data.describe()

        # Create a plot of the distribution of the target variable
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Histogram(x=self.data['target_variable']),
            row=1, col=1
        )
        fig.update_layout(
            title="Distribution of Target Variable"
        )

        # Create a scatter plot matrix of the numeric features
        numeric_features = self.data.select_dtypes([pl.Float64, pl.Int64])
        fig2 = make_subplots(
            rows=len(numeric_features.columns),
            cols=len(numeric_features.columns),
            subplot_titles=[f"{col}" for col in numeric_features.columns],
            shared_xaxes=True,
            shared_yaxes=True
        )
        for i in range(len(numeric_features.columns)):
            for j in range(len(numeric_features.columns)):
                if i == j:
                    fig2.add_trace(
                        go.Histogram(x=numeric_features.iloc[:, i]),
                        row=i+1, col=j+1
                    )
                else:
                    fig2.add_trace(
                        go.Scatter(
                            x=numeric_features.iloc[:, j],
                            y=numeric_features.iloc[:, i],
                            mode='markers'
                        ),
                        row=i+1, col=j+1
                    )
        fig2.update_layout(
            title="Scatter Plot Matrix of Numeric Features"
        )

        # Output the dashboard as an HTML file
        pio.write_html(
            fig.to_html() + fig2.to_html(),
            file='dashboard.html',
            auto_open=True
        )
