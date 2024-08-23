import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import dash
import plotly.express as px
import plotly.colors as pc
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def process_data(df):
    """
    Add new columns
    """
    df['vote_percentage'] = df['candidatevotes'] / df['totalvotes'] * 100
    df['vote_percentage'] = df['vote_percentage'].round(2)
    return df

def clean(df):
    # reduce to the main two parties
    df_majorparties = df.loc[df['party_simplified'].isin(['DEMOCRAT', 'REPUBLICAN'])]
    # issue with write ins causing duplicates
    df_majorparties = df_majorparties.loc[df_majorparties['writein']==False]
    return df_majorparties



def form_differences(df):
    df_diff = df.pivot(index=['year', 'state', 'state_po', 'state_fips'], columns='party_simplified', values='vote_percentage')
    df_diff.reset_index(inplace=True)
    df_diff['difference'] = df_diff['DEMOCRAT'] - df_diff['REPUBLICAN']
    df_diff = df_diff.loc[df_diff['state'] != 'DISTRICT OF COLUMBIA']
    return df_diff


def form_long(df):
    df_long = df.pivot(index='state', columns='year', values='difference').reset_index()
    df_long.set_index('state', inplace=True)
    return df_long
    

def cluster_pipeline(df, eps, min_samples=3):
    """
    Adjust eps and min_samples according to data
    """
    # dbscan with scaling
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Step 2: Fit DBSCAN
    dbscan = DBSCAN(eps=esps, min_samples=min_samples)  
    dbscan.fit(df_scaled)
    
    # Step 3: Get the cluster labels
    cluster_labels = dbscan.labels_
    df['cluster'] = cluster_labels
    return df


class Reshaper(pd.DataFrame):
    def piv(self, index, columns, values):
        return self.pivot(index=index, columns=columns, values=values).reset_index()


def get_color_from_scale(value, vmin, vmax, colorscale='RdBu', alpha=0.5):
    """
    Returns a color from a colorscale based on a continuous value.
    
    Parameters:
    - value: The continuous value to map to the colorscale.
    - vmin: The minimum value of the continuous scale.
    - vmax: The maximum value of the continuous scale.
    - colorscale: The name of the colorscale (e.g., 'RdBu').
    
    Returns:
    - A color in the form of a string (e.g., 'rgb(255, 0, 0)').
    """
    # Normalize the value to be between 0 and 1
    normalized_value = (value - vmin) / (vmax - vmin)
    
    # Sample the color from the colorscale
    color = pc.sample_colorscale(colorscale, [normalized_value])[0]
    rgba_color = color.replace('rgb', 'rgba').replace(')', f', {alpha})')
    return rgba_color


if __name__ == "__main__":
    df = pd.read_csv('../data/1976-2020-president.csv')
    df = process_data(df)
    df = clean(df)
    df_diff = form_differences(df)
    df_long = form_long(df_diff)
    df_clustered = cluster_pipeline(df_long, eps=1.5)
    diff_avg = df_diff.groupby(['state', 'state_po'])['difference'].mean().reset_index()
    df_clustered.to_csv('../data/clusters.csv')
    