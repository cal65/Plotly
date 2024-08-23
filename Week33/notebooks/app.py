import pandas as pd
import numpy as np
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.colors as pc
import plotly.express as px

app = Dash()


palette = ['rgb(255, 182, 193)', 'rgb(0, 0, 139)', 'rgb(139, 0, 0)', 'rgb(77,175,74)', 'rgb(205, 92, 92)', 'rgb(255, 255, 0)', 'rgb(0, 0, 255)', 'rgb(135, 206, 235)', 'rgb(139, 0, 139)']



def cluster_map(df):
    fig = px.choropleth(df,
                    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                    locationmode='USA-states',
                    locations='state_po',
                    color='cluster_name',
                    scope="usa",
                    color_discrete_sequence=palette,
                    category_orders={"cluster_name": list(cluster_name_mapping.values())},
                    title= "United Clusters"
                    )
    fig.update_layout(margin={"r":0, "l":0, "b":0}, title_x=0.5,)
    return fig


final_df = pd.read_csv('../data/clusters.csv')
clusters = final_df[['cluster', 'cluster_name']].drop_duplicates().sort_values('cluster')
cluster_dict = final_df.set_index("state")['cluster'].to_dict()
df = pd.read_csv('../data/clusters_long.csv')

cluster_name_mapping = {
    1: "Clinton Reds",
    2: "Blue Urban Northeast",
    3: "Deep Red South",
    4: "Noise",
    5: "Red Mountain Plains",
    6: "Big Sky Dakotas",
    7: "Blue Converts",
    8: "Trending Blue",
    9: "Swing States",
}


@app.callback(
    Output("Lines", "figure"),
    Input("cluster_selector", "value")
)
def cluster_line_plot(clusters=[]):
    fig = go.Figure()
    if  len(clusters) == 0:
        return fig
        
    df_cluster = df.loc[df['cluster_name'].isin(clusters)]
    
    for state in df_cluster['state'].unique():
        df_state = df.loc[df['state'] == state]
        state_color = palette[cluster_dict[state]-1]
        fig.add_trace(
        go.Scatter(
            x=df_state['year'],
            y=df_state['difference'],
            mode='lines+markers',
            line=dict(color=state_color),
            name=state,
            showlegend=True,
            )
        )
    fig.update_yaxes(range=[-50, 50])
    fig.update_layout(height=600, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)")
    return fig


# Callback to update the checklist based on the clicked region on the map
@app.callback(
    Output("cluster_selector", "value"),
    Input("choropleth_map", "clickData"),
    Input("cluster_selector", "value"),
)
def update_checklist_on_click(clickData, current_selection):
    if clickData:
        clicked_state = clickData['points'][0]['location']  # Get the clicked state code
        # Find the cluster associated with the clicked state
        selected_cluster = final_df.loc[final_df['state_po'] == clicked_state, 'cluster_name'].iloc[0]
        
        # Add or remove the cluster from the current selection
        if selected_cluster not in current_selection:
            current_selection.append(selected_cluster)
        else:
            current_selection.remove(selected_cluster)

    return current_selection


app.layout = html.Div(
    className="checklistContainer",
    children=[
        dcc.Graph(id = "choropleth_map", figure=cluster_map(final_df)),
        html.Br(),
        html.Div(
            dcc.Checklist(
                id="cluster_selector",
                options=clusters['cluster_name'],
                value = ["Swing States"],   # default value
            ),
            style={
                    'display': 'flex',
                    'justify-content': 'center',
                    'align-items': 'center',
                }
        ),
        html.Br(),
        dcc.Graph(id="Lines")
    ],
)


if __name__ == "__main__":

    app.run_server(debug=True)
