import pandas as pd
import numpy as np
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.colors as pc
import plotly.express as px
import dash_bootstrap_components as dbc

app = Dash()

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

palette = [
    "rgb(255, 182, 193)",
    "#1F46E0",
    "#A51300",  # 123
    "#1FE0B0",
    "#FF4821",
    "#FECB52",  # 456
    "#3283FE",
    "#87D1FF",
    "#AB63FA",
]  # 789


def cluster_map(df):
    fig = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locationmode="USA-states",
        locations="state_po",
        color="cluster_name",
        scope="usa",
        color_discrete_sequence=palette,
        category_orders={"cluster_name": list(cluster_name_mapping.values())},
        title="United Clusters",
    )
    fig.update_layout(
        margin={"r": 0, "l": 0, "b": 0},
        title_x=0.5,
    )
    return fig


final_df = pd.read_csv("../data/clusters.csv")
clusters = (
    final_df[["cluster", "cluster_name"]].drop_duplicates().sort_values("cluster")
)
cluster_dict = final_df.set_index("state")["cluster"].to_dict()
df = pd.read_csv("../data/clusters_long.csv")
candidates = pd.read_csv("../data/candidates.csv")


@app.callback(Output("Lines", "figure"), Input("cluster_selector", "value"))
def cluster_line_plot(clusters=[]):
    fig = go.Figure()
    if len(clusters) == 0:
        return fig

    df_cluster = df.loc[df["cluster_name"].isin(clusters)]

    for state in df_cluster["state"].unique():
        df_state = df.loc[df["state"] == state]
        state_color = palette[cluster_dict[state] - 1]
        fig.add_trace(
            go.Scatter(
                x=df_state["year"],
                y=df_state["difference"],
                mode="lines+markers",
                line=dict(color=state_color),
                name=state,
                showlegend=True,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[1976, 2020],
            y=[0, 0],
            mode="lines",
            line=dict(color="black", dash="dot"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=candidates["year"],
            y=[-45] * len(candidates),
            text=candidates["REPUBLICAN"],
            textfont=dict(
                size=14,  # Font size
                color="rgba(255, 0, 0, 0.8)",  # Text color (e.g., red with opacity)
            ),
            mode="text",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=candidates["year"],
            y=[45] * len(candidates),
            text=candidates["DEMOCRAT"],
            textfont=dict(
                size=14,  # Font size
                color="rgba(0, 0, 255, 0.8)",  # Text color (e.g., red with opacity)
            ),
            mode="text",
            showlegend=False,
        )
    )
    fig.update_yaxes(
        range=[-50, 50],
        title_text="Voting Difference",
        tickvals=[-50, -25, 0, 25, 50],  # Optional: Set specific tick values if needed
        ticktext=[
            "Republicans + 50",
            "Republicans + 25",
            "0",
            "Democrats + 25",
            "Democrats + 50",
        ],  # Optional: Custom tick labels
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# Callback to update the checklist based on the clicked region on the map
@app.callback(
    Output("cluster_selector", "value"),
    Input("choropleth_map", "clickData"),
    Input("cluster_selector", "value"),
)
def update_checklist_on_click(clickData, current_selection):
    if clickData:
        clicked_state = clickData["points"][0]["location"]  # Get the clicked state code
        # Find the cluster associated with the clicked state
        selected_cluster = final_df.loc[
            final_df["state_po"] == clicked_state, "cluster_name"
        ].iloc[0]

        # Add or remove the cluster from the current selection
        if selected_cluster not in current_selection:
            current_selection.append(selected_cluster)
        else:
            current_selection.remove(selected_cluster)

    return current_selection


app.layout = html.Div(
    className="checklistContainer",
    children=[
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        dcc.Graph(id="choropleth_map", figure=cluster_map(final_df)),
                        width=12,
                    )
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Checklist(
                                id="cluster_selector",
                                options=clusters["cluster_name"],
                                value=["Swing States"],  # default value
                                style={"width": "100%"},
                            ),
                            width=3,
                        ),
                        dbc.Col(dcc.Graph(id="Lines"), width=9),
                    ],
                    align="center",  # Optional: center-align the row's content vertically
                ),
            ],
            fluid=True,
        ),
    ],
)


if __name__ == "__main__":

    app.run_server(debug=True)
