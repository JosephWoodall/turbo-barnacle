import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd

"""
This script uses the Dash library to create a simple web application with a drop-down menu to select a metric (sales or profit) 
and a bar chart to display the selected metric. The data is read from a CSV file and the update_graph function is called when the 
user selects a different metric from the drop-down menu.

This is just a simple example and you can build upon this script to add more interactive 
elements like filters, pivot table, or other charts like line chart, scatter plot etc. and use 
different libraries like pandas and plotly to get the desired results.

It's important to note that creating a comprehensive and interactive business intelligence 
dashboard that could be substitute for powerbi and tableau is a complex task and you might 
want to consider using the above-mentioned tools or other libraries that are more suited for this purpose.
"""

app = dash.Dash()

PATH = '' # path to the data here
DASHBOARD_TITLE = '' # title of the dashboard here

df = pd.read_csv(PATH)

app.layout = html.Div([
    html.H1(DASHBOARD_TITLE),
    dcc.Dropdown(id="metric-selector", options=[
        {"label": "Sales", "value": "sales"},
        {"label": "Profit", "value": "profit"}
    ], value="sales"),
    dcc.Graph(id="metric-graph")
])

@app.callback(Output("metric-graph", "figure"), [Input("metric-selector", "value")])
def update_graph(metric):
    data = []
    if metric == "sales":
        data.append(go.Bar(x=df["month"], y=df["sales"]))
    else:
        data.append(go.Bar(x=df["month"], y=df["profit"]))
    return {"data": data}

if __name__ == "__main__":
    app.run_server(debug=True)