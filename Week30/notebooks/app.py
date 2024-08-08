# check out https://dash.plotly.com/ for documentation
# And check out https://py.cafe/maartenbreddels for more examples
from dash import Dash, Input, Output, callback, dcc, html
import pandas as pd
import plotly.express as px

app = Dash(__name__)
md = """
# Dash demo

See [The dash examples index](https://dash-example-index.herokuapp.com/) for more examples.
"""
#dtype = str is the lazy way of making sure the county fips are read ok
df = pd.read_csv('rural_download.csv', dtype=str)
df['Investment Dollars'] = df['Investment Dollars'].str.replace(',', '').astype(float)
index_cols = ['County', 'County FIPS', 'State Name']
county_investments = df.groupby(index_cols)['Investment Dollars'].sum().reset_index()
county_investments.sort_values('Investment Dollars', inplace = True)
# this fips can get cast as a number
county_investments['County FIPS'] = county_investments['County FIPS'].astype(str)
# bring in county level population estimates
pop = pd.read_csv('population_fips.csv', dtype=str)
pop['POP_ESTIMATE_2023'] = pop['POP_ESTIMATE_2023'].str.replace(',', '').astype(float)

ci_pop = pd.merge(county_investments, pop[['County FIPS', 'POP_ESTIMATE_2023']], 
                                  on='County FIPS', how='left')
ci_pop['Dollars / Capita'] = ci_pop['Investment Dollars'] / ci_pop['POP_ESTIMATE_2023']
ci_pop['Dollars / Capita'] = ci_pop['Dollars / Capita'].round(1)
# get county level investments per program area
county_program = pd.pivot_table(df, index=index_cols, 
               columns='Program Area', values='Investment Dollars').reset_index()
county_program.fillna(0, inplace=True)
ci_pop = pd.merge(ci_pop, county_program, on=index_cols, how='left')

fig = px.choropleth(
    ci_pop,
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                    locations='County FIPS',
                    color='Dollars / Capita',
                    color_continuous_scale="Viridis",
                    scope="usa",
                    labels={'Investment Dollars / Capita': 'Investment ($) / Capita'},
                    hover_data=['County', 'Investment Dollars'],
                    title='Investment Dollars / Capita per County')

features =  set(ci_pop.columns).difference(set(index_cols))                  
app.layout = html.Div([
    html.H1("Choropleth Map with Dash"),
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in ci_pop.columns if col not in index_cols],
        value='Dollars / Capita'  # default value
    ),
    dcc.RangeSlider(
        id='value-slider',
        min=ci_pop['Dollars / Capita'].min(),
        max=ci_pop['Dollars / Capita'].max(),
        value=ci_pop['Dollars / Capita'].min(),
    ),
    dcc.Graph(id='choropleth-map', figure=fig)
])


# Callback to update the slider based on the selected column
@app.callback(
    Output('value-slider', 'min'),
    Output('value-slider', 'max'),
    Output('value-slider', 'value'),
    Input('column-dropdown', 'value')
)
def update_slider(column):
    min_val = ci_pop[column].min()
    max_val = ci_pop[column].max()
    return min_val, max_val, [min_val, max_val]


# Callback to update the map based on the selected column
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('column-dropdown', 'value'), Input('value-slider', 'value')]
)
def update_choropleth_column(selected_column, selected_range):
    df = ci_pop[(ci_pop[selected_column] >= selected_range[0]) & (ci_pop[selected_column] <= selected_range[1])]
    fig = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations='County FIPS',
        color=selected_column,
        color_continuous_scale="Viridis",
        scope="usa",
        labels={'Investment Dollars / Capita': 'Investment ($) / Capita'},
        hover_data=['County', 'Investment Dollars'],
    )
    return fig

