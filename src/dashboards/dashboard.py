import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

DATA_PATH_ONE = ''
DATA_PATH_TWO = ''

# load the sample financial data
df1 = pd.read_csv(DATA_PATH_ONE)
df2 = pd.read_csv(DATA_PATH_TWO)

DF_1_X = ''
DF_1_Y = ''

DF_2_X = ''
DF_2_Y = ''

TIMESERIES_TITLE = ''
WATERFALL_TITLE = ''
FUNNEL_TITLE = ''

'''DATA ONE'''
# create the time series chart
fig_ts = px.line(df1, x=DF_1_X, y=DF_1_Y, color='Stock', title=TIMESERIES_TITLE)
fig_ts.add_trace(px.line(df2, x=DF_2_X, y=DF_2_Y, color='Stock').data[0])


# create the waterfall chart
df1_diff = df1.diff().dropna()
fig_waterfall = px.waterfall(df1_diff, x=DF_1_X, y=DF_1_Y, title=WATERFALL_TITLE)

# create the funnel chart
df1_funnel = df1[[DF_1_X, DF_1_Y]].groupby(DF_1_X).sum().reset_index()
fig_funnel = px.funnel(df1_funnel, x=DF_1_X, y=DF_1_Y, title=FUNNEL_TITLE)

# create the dash app
app = dash.Dash()
app.layout = html.Div([
    html.H1('Financial Data Dashboard'),
    html.Div([
        dcc.Dropdown(
            id='stock-dropdown',
            options=[
                {'label': 'Stock 1', 'value': 'stock1'},
                {'label': 'Stock 2', 'value': 'stock2'}
            ],
            value='stock1'
        ),
        dcc.Dropdown(
            id='chart-dropdown',
            options=[
                {'label': 'Time Series', 'value': 'timeseries'},
                {'label': 'Waterfall', 'value': 'waterfall'},
                {'label': 'Funnel', 'value': 'funnel'},
            ],
            value='timeseries'
        ),
        dcc.Graph(id='stock-graph', figure=fig_ts)
    ])
])

# update the chart based on the selected stock and chart type
@app.callback(
    dash.dependencies.Output('stock-graph', 'figure'),
    [dash.dependencies.Input('stock-dropdown', 'value'),
     dash.dependencies.Input('chart-dropdown', 'value')]
)
def update_graph(selected_stock, selected_chart):
    """

    :param selected_stock: 
    :param selected_chart: 

    """
    if selected_stock == 'stock1':
        df = df1
    else:
        df = df2
    
    if selected_chart == 'timeseries':
        return fig_ts
    elif selected_chart == 'waterfall':
        return fig_waterfall
    elif selected_chart == 'funnel':
        return fig_funnel

if __name__ == '__main__':
    app.run_server(debug=True)