import pandas as pd
import numpy as np
import plotly.express as px

def pounds_to_kg(pounds: float):
    """
    Converte libras para kilograms.
    
    Parâmetros:
    - pounds: peso em libras
    
    Retorna:
    - peso em kg
    """

    return pounds * 0.453592

def cm_to_meters(cm: int):
    """
    Converte cm para metros.
    
    Parâmetros:
    - cm: centimetros
    
    Retorna:
    - altura em metros
    """

    return cm / 100.0

def na_values_verify(df: pd.DataFrame, total = False):
    """
    Recebe um dataframe para verificar os valores nulos de todas colunas.

    Parâmetros:
    - df: dataframe a ser verificado
    - total: quantidade total de valores faltantes
    
    Retorna:
    - Informações da porcentagem de nulos por variável.
    """

    if total:
        return df.isnull().sum().sum()
    
    na_df_values = ((df.isnull().sum())/(df.shape[0]))*100
    return f'\n{"="*20} Informações NaN {"="*20}\nA porcentagem de dados faltantes por variável é:\n{na_df_values[na_df_values > 0.0]}'

def duplicated_values_verify(df: pd.DataFrame, total = False):
    """
    Recebe um dataframe para verificar os valores duplicados.
    
    Parâmetros:
    - df: dataframe a ser verificado
    - total: quantidade total de valores faltantes

    Retorna:
    - Informações da quantidade de duplicados.
    """

    if total:
        return df.duplicated().sum().sum()
    
    duplicated_df_values = df.duplicated().sum()
    return f'\n{"="*20} Informações duplicadas {"="*20}\nA porcentagem de dados duplicados é:\n{duplicated_df_values}'

def data_cleaning_hero(df_hero_info: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um dataframe para fazer a limpeza dos dados.
    
    Parâmetros:
    - df_hero_info: dataframe com as informações dos heróis

    Retorna:
    - Dataframe tratado
    """

    # Remoção da coluna Unnamed: 0
    df_hero_info = df_hero_info.drop(columns=['Unnamed: 0'])

    # Troca dos valores - por NaN
    df_hero_info = df_hero_info.replace('-', np.nan)

    # Troca de Weight e Height - por NaN
    df_hero_info['Weight'] = df_hero_info['Weight'].replace(-99.0, np.nan)
    df_hero_info['Height'] = df_hero_info['Height'].replace(-99.0, np.nan)

    # Imputação das variáveis de Weight e Height
    df_hero_info['Height'] = df_hero_info['Height'].fillna(df_hero_info.groupby(['Race','Gender'])['Height'].transform('mean'))
    df_hero_info['Weight'] = df_hero_info['Weight'].fillna(df_hero_info.groupby(['Race','Gender'])['Weight'].transform('mean'))
    df_hero_info = df_hero_info.drop('Skin color', axis=1)

    return df_hero_info

def data_cleaning_powers(df_hero_powers: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um dataframe para fazer a limpeza dos dados.
    
    Parâmetros:
    - df_hero_powers: dataframe com as informações dos poderes

    Retorna:
    - Dataframe tratado
    """

    df_hero_powers = df_hero_powers.replace('-', np.nan)
    df_hero_powers = df_hero_powers.rename(columns={'hero_names': 'name'})

    return df_hero_powers

def merge_heros(df_info: pd.DataFrame, df_poowers: pd.DataFrame):
    """
    Recebe dois dataframe para fazer o merge dos dados.
    
    Parâmetros:
    - df_powers: dataframe com as informações dos poderes
    - df_info: dataframe com as informações dos heróis

    Retorna:
    - Dataframe mergeado
    """
    df_heros = pd.merge(df_info, df_poowers, on='name')
    df_heros = df_heros.replace(0, False)
    df_heros = df_heros.replace(1, True)
    df_heros = df_heros.dropna()

    return df_heros

def pie_plot(df: pd.DataFrame, coluna: str, titulo = str):
    """
    Recebe um dataframe para fazer a plotagem dos dados.
    
    Parâmetros:
    - df: dataframe com as informações
    - coluna: nome da coluna
    - titulo: titulo desejado

    Retorna:
    - Gráfico de pizza
    """

    data = df[coluna].value_counts().reset_index()
    data.columns = [coluna, 'count']  

    fig = px.pie(data, names=coluna, values='count', color_discrete_sequence=['#05327a','#1f77b4'],
                hole=0.3, labels={coluna: coluna}, title=titulo)

    fig.update_traces(textfont_color='white', textinfo='percent+label', textfont_size=18)
    fig.update_layout(
        showlegend=False,
        title=dict(text=titulo, x=0.5, xanchor='center', font=dict(size=22, color='white')),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)', 
        font=dict(color='white')  
    )

    return fig

def scatter_plot(df: pd.DataFrame, x_axis: str, y_axis: str, cor: str, titulo: str, largura: int = 700):
    """
    Recebe um dataframe para fazer a plotagem dos dados.
    
    Parâmetros:
    - df: dataframe com as informações
    - x_axis: nome da coluna do eixo X
    - y_axis: nome da coluna do eixo Y
    - cor: nome da cor
    - titulo: título desejado
    - largura: largura do gráfico

    Retorna:
    - Gráfico de dispersão
    """


    fig = px.scatter(df, x=x_axis, y=y_axis, color=cor, symbol=cor,
                    color_discrete_map={'Male': 'white', 'Female': '#05327a', 'good': 'white', 'bad': 'red', 'neutral': '#05327a'})

    fig.update_traces(textfont_size=18)
    fig.update_layout(
        showlegend=True,
        legend_title_text='',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3, 
            xanchor="center",
            x=0.5,
            font=dict(size=22, color='white'),
        ),
        title=dict(text=titulo, x=0.5, xanchor='center', font=dict(size=22, color='white')),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)', 
        font=dict(color='white'),
        width=largura, 
        height=500
    )

    return fig

def bar_plot(df: pd.DataFrame, x_axis: str, y_axis: str, cor: str, titulo: str, modo: str = 'group', largura: int = 700):
    """
    Recebe um dataframe para fazer a plotagem dos dados.
    
    Parâmetros:
    - df: dataframe com as informações
    - x_axis: nome da coluna do eixo X
    - y_axis: nome da coluna do eixo Y
    - cor: nome da cor
    - titulo: título desejado
    - modo: modo das barras
    - largura: largura do gráfico

    Retorna:
    - Gráfico de dispersão
    """

    color_map = {True: 'white', False: '#327fa8'}  
    fig = px.bar(df, x=x_axis, y=y_axis, color=cor,
                 barmode=modo, title=titulo,
                 color_discrete_map=color_map)
    fig.update_traces(
        texttemplate='%{y}', textposition='outside',
        marker_line_color='rgb(8,48,107)', marker_line_width=1.5,
        textfont=dict(size=18, color='white')  # Ajusta o tamanho e a cor da fonte dos valores acima das barras
    )
    fig.update_layout(
        showlegend=True,
        legend_title_text='',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=22, color='white'),
        ),
        title=dict(text=titulo, x=0.5, xanchor='center', font=dict(size=22, color='white')),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        width=largura,
        height=600,
        xaxis=dict(
            title=dict(text='', font=dict(size=22, color='white')),  
            tickfont=dict(size=22, color='white') 
        )
    )

    return fig