#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import plotly.express as px

def twod_visualization(df):
    fig = go.Figure()

    data = df.iloc[:, -1]
    targets = data.unique()

    colors = ['rgb(255, 204, 0)', 'rgb(255, 100, 0)']  # Specify colors for each class

    for target, color in zip(targets, colors):
        indicesToKeep = (data == target)
        fig.add_trace(go.Scatter(
            x=df.loc[indicesToKeep, 'PC1'],
            y=df.loc[indicesToKeep, 'PC2'],
            mode='markers',
            name=str(target),
            marker=dict(size=8, color=color, opacity=0.7)
        ))

    fig.update_layout(
        xaxis=dict(title='Principal Component 1', tickfont=dict(size=10)),
        yaxis=dict(title='Principal Component 2', tickfont=dict(size=10)),
        title='2 component PCA',
        width=600,
        height=600,
        showlegend=True,
        legend=dict(font=dict(size=8)),
        margin=dict(l=40, r=40, t=40, b=40)  # Add margin for better spacing
    )
    fig.update_traces(marker=dict(size=8))  # Adjust marker size

    st.plotly_chart(fig)

    
def threed_visualization(df):
    fig = go.Figure()

    data = df.iloc[:, -1]
    targets = data.unique()

    colors = ['rgb(255, 204, 0)', 'rgb(255, 100, 0)']  # 각 클래스에 대한 색상 지정

    for target, color in zip(targets, colors):
        indicesToKeep = (data == target)
        fig.add_trace(go.Scatter3d(
            x=df.loc[indicesToKeep, 'PC1'],
            y=df.loc[indicesToKeep, 'PC2'],
            z=df.loc[indicesToKeep, 'PC3'],
            mode='markers',
            name=str(target),
            marker=dict(size=3, color=color)
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Principal Component 1'),
            yaxis=dict(title='Principal Component 2'),
            zaxis=dict(title='Principal Component 3')
        ),
        title='3 component PCA',
        width=800,
        height=800,
        showlegend=True,
        legend=dict(font=dict(size=8)),
    )

    st.plotly_chart(fig)