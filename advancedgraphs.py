import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import folium
from streamlit_folium import folium_static
import networkx as nx
from pyvis.network import Network
import plotly.express as px
import time

def main():
    st.set_page_config(page_title='Interactive Data Visualizations', layout='wide')
    st.title("📊 Interactive Data Visualizations")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        # Sidebar options
        viz_option = st.sidebar.selectbox("Choose a Visualization", [
            "3D Surface Plot", "Correlation Heatmap", "Mapping Locations", "Animated Line Chart", "Social Network Graph"])
        
        if viz_option == "3D Surface Plot":
            plot_3d_surface(df)
        elif viz_option == "Correlation Heatmap":
            plot_correlation_heatmap(df)
        elif viz_option == "Mapping Locations":
            plot_map(df)
        elif viz_option == "Animated Line Chart":
            plot_animated_line(df)
        elif viz_option == "Social Network Graph":
            plot_social_graph(df)
        

def plot_3d_surface(df):
    st.subheader("🌄 3D Surface Plot")
    if df.shape[1] < 3:
        st.error("Dataset must have at least 3 numerical columns.")
        return
    x_col, y_col, z_col = st.selectbox("X-axis", df.columns), st.selectbox("Y-axis", df.columns), st.selectbox("Z-axis", df.columns)
    
    fig = go.Figure(data=[go.Surface(z=df[z_col].values.reshape((len(df[x_col].unique()), len(df[y_col].unique()))), x=df[x_col], y=df[y_col])])
    fig.update_layout(title="3D Surface Plot", scene=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig)

def plot_correlation_heatmap(df):
    st.subheader("🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.error("No numerical columns found.")
        return
    
    corr = numeric_df.corr()
    fig = sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
    st.pyplot(fig.get_figure())

def plot_map(df):
    st.subheader("🗺️ Mapping Locations")
    if not {'latitude', 'longitude'}.issubset(df.columns):
        st.error("Dataset must have 'latitude' and 'longitude' columns.")
        return
    
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5)
    for _, row in df.iterrows():
        folium.Marker([row['latitude'], row['longitude']], popup=str(row.get('name', 'Location'))).add_to(m)
    folium_static(m)

def plot_animated_line(df):
    st.subheader("📈 Animated Line Chart")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.error("Dataset must have at least two numerical columns.")
        return
    
    x_col, y_col = st.selectbox("X-axis", numeric_df.columns), st.selectbox("Y-axis", numeric_df.columns)
    
    fig = px.line(df, x=x_col, y=y_col, animation_frame=df.index, title="Animated Line Chart", template='plotly_dark')
    st.plotly_chart(fig)

def plot_social_graph(df):
    st.subheader("👥 Social Network Graph")
    if df.shape[1] < 2:
        st.error("Dataset must have at least two columns (Source and Target).")
        return
    
    src_col, tgt_col = st.selectbox("Source Column", df.columns), st.selectbox("Target Column", df.columns)
    
    G = nx.from_pandas_edgelist(df, source=src_col, target=tgt_col)
    nt = Network(notebook=False, height="500px", bgcolor="#222222", font_color="white")
    nt.from_nx(G)
    nt.show("graph.html")
    st.components.v1.html(open("graph.html").read(), height=550)

if __name__ == "__main__":
    main()
