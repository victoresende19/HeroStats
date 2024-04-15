import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import plotly.express as px

@st.cache_data(show_spinner=False, ttl=24*3600, max_entries=2)
def pca(df: pd.DataFrame):
    """
    Recebe um dataframe para fazer a redução de dimensionalidade dos dados.
    
    Parâmetros:
    - df: dataframe com as informações

    Retorna:
    - Dados PCA
    """

    pca = PCA(0.8, random_state=42)
    data_pca = pca.fit_transform(df)
    return data_pca

@st.cache_data(show_spinner=False, ttl=24*3600, max_entries=2)
def find_optimal_clusters(data, min_clusters, max_clusters):
    """
    Esta função calcula o KMeans para diferentes quantidades de clusters, de 'min_clusters' até 'max_clusters',
    e retorna o número de clusters que maximiza o silhouette score.

    Parâmetros:
    - data: conjunto de dados (DataFrame ou array NumPy) para clusterização.
    - min_clusters: mínimo de clusters a ser testado.
    - max_clusters: máximo de clusters a ser testado.

    Retorna:
    - Um dicionário contendo o número ótimo de clusters e o respectivo silhouette score.
    """
    
    silhouette_scores = []
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append((n_clusters, silhouette_avg))
    
    optimal_n_clusters, best_score = max(silhouette_scores, key=lambda x: x[1])
    
    return optimal_n_clusters, best_score

@st.cache_data(show_spinner=False, ttl=24*3600, max_entries=2)
def kmeans(num_clusters: int, data_pca: list, df_powers: pd.DataFrame):
    """
    Esta função faz a aplicação do algoritmo KMeans utilizando o número de clusters ideiais
    e os dados necessário.

    Parâmetros:
    - data_pca: conjunto de dados PCA para clusterização.
    - df_powers: conjunto de dados dos poderes para criar nova coluna com as labels
    - num_clusters: número ideal de clusters.

    Retorna:
    - Um dicionário contendo o número ótimo de clusters e o respectivo silhouette score.
    """

    km_model = KMeans(num_clusters, random_state=42, n_init=10)
    km_model.fit(data_pca)
    df_powers['Cluster'] = km_model.labels_

    return df_powers

@st.cache_data(show_spinner=False, ttl=24*3600, max_entries=2)
def TSNE_plot(data_pca: list, df: pd.DataFrame, titulo: str, largura: int = 700):
    """
    Esta função faz a plotagem dos dados usando t-SNE para visualização dimensional reduzida,
    e utiliza Plotly para criar um gráfico de dispersão interativo.

    Parâmetros:
    - data_pca: conjunto de dados PCA para clusterização.
    - df: conjunto de dados após o kmeans com a coluna Cluster.
    - titulo: título desejado para o gráfico.
    - largura: largura do gráfico.

    Retorna:
    - Uma visualização TSNE interativa usando Plotly.
    """

    tsne_clusters = TSNE(verbose=1, random_state=42)
    tsne_results = tsne_clusters.fit_transform(data_pca)
    
    df['t-SNE-1'] = tsne_results[:, 0]
    df['t-SNE-2'] = tsne_results[:, 1]

    color_map = {0: 'Cluster 0', 1: 'Cluster 1'} 
    df['color'] = df['Cluster'].map(color_map)

    fig = px.scatter(df, x='t-SNE-1', y='t-SNE-2', color='color', symbol='color', 
                    color_discrete_map={'Cluster 0': 'white', 'Cluster 1': '#f55142'}, 
                     title=titulo, width=largura, height=500, custom_data=["name", 'Cluster'])

    fig.update_traces(
        marker=dict(size=18, line=dict(width=2)),
        hovertemplate='<br>'.join([
            'Nome: %{customdata[0]}',
            'Cluster: %{customdata[1]}',
        ])
    )
    fig.update_layout(
        showlegend=True,
        legend_title_text='',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5,
            font=dict(size=22, color='white')
        ),
        title=dict(font=dict(size=22, color='white')),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white')
    )

    return fig