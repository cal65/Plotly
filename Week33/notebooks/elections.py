import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import dash
import plotly.express as px


def process_data(df):
    df['vote_percentage'] = df['candidatevotes'] / df['totalvotes'] * 100
    df['vote_percentage'] = df['vote_percentage'].round(2)

df_majorparties = df.loc[df['party_simplified'].isin(['DEMOCRAT', 'REPUBLICAN'])]