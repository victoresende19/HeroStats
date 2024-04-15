import streamlit as st
import pandas as pd
import numpy as np
from utils.style import style
from utils.eda import data_cleaning_hero, data_cleaning_powers, pie_plot, scatter_plot, bar_plot, merge_heros, pounds_to_kg, cm_to_meters
from utils.cluster import pca, find_optimal_clusters, kmeans, TSNE_plot
from utils.classifier import classifier
from utils.regressor import regressor, features

st.set_page_config(layout="wide", page_icon='🦸', page_title='HeroStats')
st.markdown(style(), unsafe_allow_html=True)
st.markdown("<h1 style='text-align: left; font-size:52px; color: white'>HeroStats</h1>",unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size:16px'>A HeroStats é uma instituição de heróis a qual visa consolidar heróis durante o mundo. Nossa missão é utilizar dados e técnicas estatísticas para melhores tomadas de decisão, seja para estratégias de ação, contratação ou gerenciamento. Explore as abas e entenda nossas análises!</p><br><br>", unsafe_allow_html=True)
eda, cluster, align, weigth = st.tabs(["Exploração dos dados", "Formação de equipes", "Alinhamento", "Previsão do peso"])


df_hero_info = pd.read_csv('data/heroes_information.csv')
df_hero_powers = pd.read_csv('data/super_hero_powers.csv')

df_hero_info = data_cleaning_hero(df_hero_info)
df_hero_powers = data_cleaning_powers(df_hero_powers)

df_heros = merge_heros(df_hero_info, df_hero_powers)
df_heros['IMC'] = (df_heros['Weight']/(df_heros['Height']/100)**2)

with eda:
    st.write('')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quantidade de heróis", len(df_hero_info))
    col2.metric("Altura média", f"{int(df_hero_info.Height.mean())/100} metros")
    col3.metric("Peso média", f"{round(df_hero_info.Weight.mean(), 2)} kg")
    col4.metric("Raça mais comum", f"{df_hero_info.Race.mode().values[0]}")

    st.write('')
    st.write('')
    st.write('')

    with st.form(key='my_form_map'):
        with st.expander("Faça filtros para encontrar seu herói preferido!"):
            col1, col2, col3 = st.columns(3)

            with col1:
                names = st.multiselect('Nome', df_hero_info.name.unique())
                race = st.multiselect('Raça', df_hero_info.Race.unique())
            with col2:
                gender = st.multiselect('Gênero', df_hero_info.Gender.unique())
                publisher = st.multiselect('Editora', df_hero_info.Publisher.unique())
            with col3:
                eye_color = st.multiselect('Cor dos olhos', df_hero_info['Eye color'].unique())
                alignment = st.multiselect('Alinhamento', df_hero_info.Alignment.unique())

        submit_button = st.form_submit_button(label='Encontrar herói 💥')


        if submit_button:
            if not any([names, race, gender, publisher, eye_color, alignment]):
                st.table(df_hero_info.head(5))
            else:
                filtered_df = df_hero_info[
                    (df_hero_info['name'].isin(names) if names else True) &
                    (df_hero_info['Race'].isin(race) if race else True) &
                    (df_hero_info['Gender'].isin(gender) if gender else True) &
                    (df_hero_info['Publisher'].isin(publisher) if publisher else True) &
                    (df_hero_info['Eye color'].isin(eye_color) if eye_color else True) &
                    (df_hero_info['Alignment'].isin(alignment) if alignment else True)
                ]

                st.table(filtered_df)
        

    st.markdown("<p style='text-align: left; font-size:16px'>É comumente divulgado que vilões normalmente são altos ou possuem grande peso, temos como provar isso? A HeroStats demonstra abaixo essa relação! Inicialmente as respectivas porcentagens dos gêneros e alinhamento. Então, a relação dos pesos e alturas pelos gêneros e alinhamentos.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(pie_plot(df_hero_info, 'Gender', 'Gêneros'))
        st.plotly_chart(scatter_plot(df_hero_info, 'Height', 'Weight', cor='Gender', titulo='Avaliação dos peso e altura por gênero'))
    with col2:
        st.plotly_chart(pie_plot(df_hero_info, 'Alignment', 'Alinhamento'))
        st.plotly_chart(scatter_plot(df_hero_info, 'Height', 'Weight', cor='Alignment', titulo='Avaliação dos peso e altura por alinhamento'))

    st.markdown("<h1 style='text-align: left; font-size:52px; color: white'>Poderes</h1>",unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; font-size:16px'>O que torna uma pessoa comum em herói? A capacidade de salvar ou proteger outras pessoas? Ter poderes? Isso! Porém a grande diferença esta no uso desse poder, para o bem ou mau. Assim, abaixo estão alguns pontos interessantes sobre os poderes dos super-heróis. </p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(pie_plot(df_hero_powers, 'Immortality', 'Imortalidade'))
       
    with col2:
        st.plotly_chart(pie_plot(df_hero_powers, 'Invisibility', 'Invisibilidade'))
    
    df_count = df_heros.groupby(['Alignment', 'Immortality']).size().reset_index(name='Count')
    st.plotly_chart(bar_plot(df_count, 'Alignment', 'Count', 'Immortality', 'Alinhamento por imortalidade', largura=1750))



with cluster:
    data_pca = pca(df_hero_powers.drop('name', axis=1))
    num_clusters, best_score = find_optimal_clusters(data_pca, 2, 10)
    df_hero_powers = kmeans(num_clusters, data_pca, df_hero_powers)

    perc_cluster_0 = (len(df_hero_powers[df_hero_powers.Cluster == 0])/len(df_hero_powers))*100
    perc_cluster_1 = (len(df_hero_powers[df_hero_powers.Cluster == 1])/len(df_hero_powers))*100

    st.write('')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quantidade de equipes", num_clusters)
    col2.metric("Porcentagem no cluster 1", f"{round(perc_cluster_0, 2)} %")
    col3.metric("Porcentagem no cluster 2", f"{round(perc_cluster_1, 2)} %")
    col4.metric("Melhor score - Sillhoute score", f"{round(best_score, 2)}")

    st.write('')
    st.write('')
    st.write('')
    st.markdown("<p style='text-align: left; font-size:16px'>Atualmente, a HeroStats trabalha para fazer as melhores divisões de equipes. Assim, visando as características dos poderes e a sinergia entre os membros, decidiu-se aplicar um algoritmo de clusterização, o KMeans, aliado ao método de silhueta para encontrar a melhor quantidade de equipes. Assim, teremos equipes das quais os membros terão a melhor sinergia possível entre si. </p>", unsafe_allow_html=True)
    
    st.plotly_chart(TSNE_plot(data_pca, df_hero_powers, 'Distribuição dos clusters', 1750))

    st.markdown("<p style='text-align: left; font-size:16px'>Agora com as equipes formadas, os heróis tem mais liberdade e sinergia para que possam atuar cada vez mais. Mas, já pensou na possibilidade de descobrir em qual equipe seu herói foi alocado? Teste os filtros abaixo! </p>", unsafe_allow_html=True)

    with st.form(key='cluster'):
        with st.expander("Faça filtros para saber em qual equipe seu herói foi alocado!"):
            col1, col2 = st.columns(2)
            with col1:
                names = st.multiselect('Nome', df_hero_powers.name.unique())
            with col2:
                clusters = st.multiselect('Cluster', df_hero_powers.Cluster.unique())

        submit_button = st.form_submit_button(label='Encontrar herói 💥')


        if submit_button:
            if not any([names, clusters]):
                st.table(df_hero_powers.head(5))
            else:
                filtered_df = df_hero_powers[
                    (df_hero_powers['name'].isin(names) if names else True) &
                    (df_hero_powers['Cluster'].isin(clusters) if clusters else True)
                ]

                st.table(filtered_df)

    
with align:
    df_heros =  df_heros[df_heros['Alignment'] != 'neutral']
    precision, recall, f1score, acc, classifier = classifier(df_heros)

    st.write('')
    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia", round(acc, 2))
    col2.metric("F1-Score", round(f1score, 2))
    col3.metric("Precisão", round(precision, 2))

    st.markdown("<p style='text-align: left; font-size:16px'>Imagine, uma pessoa com super poderes nova aparece e não está mapeada na base de dados da HeroStats. Teria como prevermos o alinhamento, de acordo com as características, em bom ou mau? Claro! Abaixo você pode testar o algoritmo Naive Bayes, que faz a previsão de alinhamento. </p>", unsafe_allow_html=True)

    
    with st.form(key='align'):
        with st.expander("Preveja o alinhamento através das características! ", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                imc = st.number_input('IMC')
                marksmanship = st.selectbox('Boa pontaria', df_heros.Marksmanship.unique())
            with col2:
                durability = st.selectbox('Durabilidade', df_heros.Durability.unique())
                insanity = st.selectbox('Insano', df_heros.Insanity.unique())
            with col3:
                stealth = st.selectbox('Furtivo', df_heros.Stealth.unique())
                cryokinesis = st.selectbox('Criocinese (Redução da temperatura)', df_heros.Cryokinesis.unique())

        submit_button = st.form_submit_button(label='Prever alinhamento 💥')

        if submit_button:
            if not any([imc, marksmanship, durability, insanity, stealth, cryokinesis]):
                st.warning('Preencha os campos, por favor', icon="⚠️")
            else:
                align = np.array([[durability, stealth, marksmanship, cryokinesis, insanity, imc]])
                alignment_predicted = classifier.predict(align)
                
                map_alignment = {'good': 'bom', 'bad': 'mau'}
                st.markdown(f"<p style='text-align: left; font-size:16px'>O alinhamento de um herói com essas características é: <strong> {map_alignment[alignment_predicted[0]]}</strong></p>", unsafe_allow_html=True)

    
with weigth:
    r2, mape, mae, regressor = regressor(df_heros)

    st.write('')
    col1, col2, col3 = st.columns(3)
    col1.metric("R2", round(r2, 2))
    col2.metric("Porcentagem de erro (MAPE)", f"{round(mape, 2)}%")
    col3.metric("Erro absoluto (MAE)", round(mae, 2))

    st.markdown("<p style='text-align: left; font-size:16px'>Por fim, visando a criação de uniformes e estratégias de ação dos heróis, foi solicitado à HeroStats conseguir prever o peso dos heróis a partir de determinadas características. Assim, foi utilizada a técnica de regressão com Random Forest por conta dos diversos outliers contidos na base, principalmente para os dados de peso. Além disso, para escolher a melhor característica a fim de prever o peso, utilizou-se a técnica de Feature Importance, do qual evidenciou o IMC (criado no processamento dos dados) como melhor preditora do peso. A seguir é possível testar o algoritmo e prever o peso dos heróis!</p>", unsafe_allow_html=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    features(df_heros)

    with st.form(key='weight'):
        with st.expander("Preveja o peso! ", expanded=True):
            imc = st.number_input('IMC')
                
        submit_button = st.form_submit_button(label='Prever alinhamento 💥')

        if submit_button:
            if not any([imc]):
                st.warning('Preencha o campo, por favor', icon="⚠️")
            else:
                align = np.array([[imc]])
                weight_predicted = regressor.predict(align)
                st.markdown(f"<p style='text-align: left; font-size:16px'>O alinhamento de um herói com essas características é: <strong> {round(pounds_to_kg(weight_predicted[0]), 2)} kg</strong></p>", unsafe_allow_html=True)
