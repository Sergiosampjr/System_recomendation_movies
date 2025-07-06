import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import streamlit as st
from sklearn.metrics import silhouette_score,calinski_harabasz_score, davies_bouldin_score


print('Pré-processamento do dataset')
df = pd.read_csv('netflix_titles.csv')
print(f'Valores faltantes \n {df.isnull().sum()} ')
print(f'Às 5 primeiras linhas do dataset \n {df.head()}')

print(f'Descrições do dataset:\n {df.describe()}.')
print(f'Tipos de colunas do dataset:\n {df.dtypes}')
#verificando possíveis duplicadas
print(f'Soma de duplicadas:{df.duplicated().sum()}')
#mostrar_tudo = pd.set_option('display.max_colums',None)
#print(mostrar_tudo)
print(df['date_added'].dropna().unique()[:10])  # Mostra os 10 primeiros valores únicos sem nulos   
print(df)
st.title('Análise e exploração do dataset')
st.dataframe(df)

'''
colunas para remover
'''
# A. Imputar colunas com muitos valores faltantes
fill_values = {'director': 'Unknown', 'cast': 'Unknown', 'country': 'Unknown'}
df.fillna(value=fill_values, inplace=True)

# B. Remover linhas com poucos valores faltantes nas colunas críticas
df.dropna(subset=['date_added', 'rating', 'duration'], inplace=True)

# Verificar se o tratamento funcionou
print("Valores faltantes após o tratamento:")
print(df.isnull().sum())

df['added_year'] = pd.to_datetime(df['date_added'].str.strip(), format="%B %d, %Y", errors='coerce').dt.year
df['added_month'] = pd.to_datetime(df['date_added'].str.strip(), format="%B %d, %Y", errors='coerce').dt.month



# 2.2 Tratar 'duration' (você já começou esta parte)
# Vamos criar duas colunas para diferenciar filmes de séries
df['duration_min'] = df.apply(
    lambda row: int(row['duration'].split()[0]) if row['type'] == 'Movie' else 0,
    axis=1
)
df['duration_seasons'] = df.apply(
    lambda row: int(row['duration'].split()[0]) if row['type'] == 'TV Show' else 0,
    axis=1
)

# 2.3 Remover colunas originais e não úteis para o modelo
df_processed = df.drop([
    'show_id', 'title', 'description', # Identificadores e texto livre
    'date_added', 'duration',          # Colunas originais já transformadas
    'director', 'cast', 'country', 'listed_in' # Colunas complexas (para um primeiro modelo)
], axis=1)

print("\nDataFrame pronto para a etapa final de pré-processamento:\n")
print(df_processed.head())


# 3.1 Identificar as colunas finais
numeric_features = ['release_year', 'added_year', 'added_month', 'duration_min', 'duration_seasons']
categorical_features = ['type', 'rating']

# 3.2 Criar o pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop' # Descarta colunas não especificadas
)

# 3.3 (Opcional, mas recomendado) Criar um Pipeline completo com um modelo
# Isso garante que todo o pré-processamento seja feito de forma consistente
# Vamos usar KMeans (clusterização) como um exemplo de tarefa não supervisionada
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clusterer', KMeans(n_clusters=3, random_state=42, n_init='auto'))
])

# 3.4 Treinar o pipeline
full_pipeline.fit(df_processed)

# Agora o pipeline está treinado. Você pode usá-lo para ver os clusters, por exemplo:
# labels = full_pipeline.named_steps['clusterer'].labels_
# print("\nPrimeiros 10 labels de cluster:", labels[:10])

# Se você só quiser os dados transformados:
X_final = preprocessor.fit_transform(df_processed)
print(f"\nShape final dos dados para o modelo: {X_final.shape}")

st.dataframe(df)

df['cluster'] = full_pipeline.named_steps['clusterer'].labels_

inertias = []
silhouettes = []
k_range = range(2, 15)

for k in k_range:
    kmeans = KMeans(n_clusters=k,random_state=42,n_init='auto')
    x_transformed = preprocessor.fit_transform(df_processed)
    kmeans.fit(x_transformed)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(x_transformed, kmeans.labels_))

st.subheader("Escolha automática do número ideal de clusters (Silhouette Score)")

# Transformar os dados
X_transformed = preprocessor.fit_transform(df_processed)

# Faixa de k a testar
k_range = range(2, 15)
silhouettes = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_transformed)
    score = silhouette_score(X_transformed, kmeans.labels_)
    silhouettes.append(score)


df['cluster'] = full_pipeline.named_steps['clusterer'].labels_
st.write(df.groupby('cluster').mean(numeric_only=True))
st.write(df.groupby('cluster')['type'].value_counts())
st.write(df.groupby('cluster')['rating'].value_counts())



# Encontrar o melhor k
melhor_k = k_range[np.argmax(silhouettes)]
st.write(f"🔍 Melhor número de clusters com base no Silhouette Score: **{melhor_k}**")

# Plotar o gráfico
fig, ax = plt.subplots()
ax.plot(k_range, silhouettes, marker='o', color='green')
ax.axvline(x=melhor_k, color='red', linestyle='--', label=f'Melhor k = {melhor_k}')
ax.set_title('Silhouette Score por número de clusters')
ax.set_xlabel('Número de Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.legend()
st.pyplot(fig)


# Plotando o método do cotovelo
plt.figure(figsize=(10, 5))
plt.plot(k_range, inertias, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.grid(True)
plt.show()


pca = PCA(n_components=2)
X_pca = pca.fit_transform(preprocessor.fit_transform(df_processed))

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='tab10', alpha=0.6)
plt.title('Visualização dos clusters com PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')
plt.show()

X_transformed = preprocessor.fit_transform(df_processed)
X_pca = PCA(n_components=2).fit_transform(X_transformed)

df['cluster'] = full_pipeline.named_steps['clusterer'].labels_

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='tab10', alpha=0.6)
plt.title('Visualização dos clusters com PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')
st.pyplot(plt)

st.subheader("Resumo por cluster")
st.write(df.groupby('cluster').agg({
    'release_year': 'mean',
    'duration_min': 'mean',
    'duration_seasons': 'mean',
    'type': lambda x: x.value_counts().index[0],
    'rating': lambda x: x.value_counts().index[0]
}))


selected_cluster = st.selectbox("Selecione um cluster para explorar", sorted(df['cluster'].unique()))
st.write(df[df['cluster'] == selected_cluster])

st.text('Métricas de avaliação do modelo:')
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

X_transformed = preprocessor.fit_transform(df_processed)
labels = full_pipeline.named_steps['clusterer'].labels_

print("Silhouette Score:", silhouette_score(X_transformed, labels))
print("Calinski-Harabasz Score:", calinski_harabasz_score(X_transformed, labels))
print("Davies-Bouldin Score:", davies_bouldin_score(X_transformed, labels))

st.subheader("🎬 Sistema de Recomendação por Cluster")


# Lista de títulos disponíveis
titulos_disponiveis = df['title'].dropna().unique()
titulo_escolhido = st.selectbox("Escolha um título que você já assistiu:", sorted(titulos_disponiveis))

# Encontrar o cluster do título escolhido
cluster_do_titulo = df[df['title'] == titulo_escolhido]['cluster'].values[0]

# Filtrar outros títulos do mesmo cluster (excluindo o próprio)
recomendacoes = df[(df['cluster'] == cluster_do_titulo) & (df['title'] != titulo_escolhido)]

# Mostrar recomendações
st.write(f"🔍 Títulos semelhantes a **{titulo_escolhido}** (Cluster {cluster_do_titulo}):")
st.dataframe(recomendacoes[['title', 'type', 'release_year', 'rating']].sample(5))

