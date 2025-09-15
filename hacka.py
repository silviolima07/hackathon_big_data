# -*- coding: utf-8 -*-
"""

# Colunas do dataset de teste:

 - semana (número inteiro): número da semana (1 a 4 de janeiro/2023)
 - pdv (número inteiro): código do ponto de venda
 - produto (número inteiro): código do SKU
 - quantidade (número inteiro): previsão de vendas
"""

!pip install parquet -q
!pip install pyarrow -q

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
import pyarrow as pa # Import pyarrow core library

pd.set_option('display.max_columns', 60)
pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
# %matplotlib inline

plt.rcParams['font.size'] = 24

from IPython.core.pylabtools import figsize

import seaborn as sns
sns.set(font_scale = 2)

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer 
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer




"""# Modelo deve prever a quantidade com estas colunas no dataset df_teste, gerando a coluna quantidade"""

data = {'semana': [1,2,3, 4, 5],
        'pdv': [1023, 1045, 1023, 1088, 1010],
        'produto': [123,234, 456, 123, 550]}
df_teste = pd.DataFrame(data)

def missing_values_table(df):
        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

        print ("Seu dataframe tem " + str(df.shape[1]) + " colunas.\n"
            "Há " + str(mis_val_table_ren_columns.shape[0]) +
              " colunas que possuem valores ausentes.")

        return mis_val_table_ren_columns

path='/content/drive/MyDrive/HACKA/'

"""# Ler os arquivos parquet com  pyarrow
- cada parte do arquivo parquet foi lida individualmente
- depois foram concatenadas
- a quantidade de linhas e colunas ficou igual a leitura com duckdb
"""

# Get the directory of the current file
directory = os.path.dirname(path)

# List all parquet files in the directory that are part of the dataset
all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith('part-') and f.endswith('.parquet')]

#print("Arquivos Parquet encontrados:")
#for file in all_files:
    #print(file)
print(f'Lendo e concatenando registros...')

tables = []
for file in all_files:
    try:
        # Read each parquet file into a PyArrow table
        table = pq.read_table(file)
        tables.append(table)
        print("\nTables Pyarrow:\n", tables)
    except Exception as e:
        print(f"Error reading file {file}: {e}")

if tables:
    # Concatenate the PyArrow tables
    # Use the promote parameter to handle schema differences by promoting to a common type
    combined_table = pa.concat_tables(tables, promote=True)

    # Convert the combined PyArrow Table to a Pandas DataFrame
    df_pyarrow_combined = combined_table.to_pandas()

    print("\nDataset pandas lido e concatenado pelo PyArrow (todas as colunas):")
    print("Colunas:", df_pyarrow_combined.columns.tolist())
    print(df_pyarrow_combined.shape)
else:
    print("No tables were read.")

df_pyarrow_combined.columns

df_pyarrow_combined.head()




"""# Coluna e tipos esperados no dataset de teste e submissão
- coluna semana no ano (1 a 52)
- pdv (número inteiro): código do ponto de venda
- produto (número inteiro): código do SKU
- quantidade (número inteiro): previsão de vendas
"""

df = df_pyarrow_combined.copy()

"""# Estatistica Descritiva"""

df.describe().T

"""# Tipo de cada coluna"""

df.info()

"""# Percentual de dados ausentes por coluna"""

missing_values_table(df)

"""# Colunas pdv e produto apresentam 99% de dados ausentes.
# Serao substituidas por
- internal_store_id
- internal_product_id
"""

colunas = ['internal_store_id',  'pdv',  'internal_product_id', 'produto', 'transaction_date','quantity']

missing_values_table(df[colunas])

"""# Após  remover as linhas ausentes em quantity as linhas de outras colunas também foram removidas."""

df = df.dropna(subset=["quantity"]).copy()
missing_values_table(df[colunas])

df.columns

df2 = df[['internal_store_id', 'internal_product_id', 'transaction_date', 'quantity']].copy()

"""# Coluna são renomeadas"""

df2.rename(columns={'internal_store_id': 'pdv', 'internal_product_id': 'produto', 'quantity': 'quantidade'}, inplace=True)

df2.isna().sum()

"""# Extrair o numero da semana no ano a partir da data de transacao"""

df2['transaction_date'] = pd.to_datetime(df2['transaction_date'])
df2['semana'] = df2['transaction_date'].dt.isocalendar().week
del df2['transaction_date']

df2.columns

df2

missing_values_table(df2)

df2.info()

df2

df2.nunique()

"""# Coluna quantidade tem valores negativos
# Outlier certamente.
"""

df2.describe()

"""# Remover Outliers"""

df_com_outliers = df2.copy()
df_com_outliers.shape

df_com_outliers.head(3)

df_com_outliers.describe()

"""# Modo padrão usando IQT,  Q1 e Q3"""

def remove_outliers(df):
    print("Antes:", df.shape)
    first_quartile = df_com_outliers['quantidade'].describe()['25%']
    third_quartile = df_com_outliers['quantidade'].describe()['75%']

    iqr = third_quartile - first_quartile

    df_sem_outliers= df_com_outliers[(df_com_outliers['quantidade'] > (first_quartile - 1.5 * iqr)) &
            (df_com_outliers['quantidade'] < (third_quartile + 1.5 * iqr))].copy()
            
    # Remover valores negativos        
    df_sem_outliers = df_sem_outliers[df_sem_outliers['quantidade'] >= 0]
    print("Depois:", df_sem_outliers.shape)
    return df_sem_outliers

df_sem_outliers = remove_outliers(df_com_outliers)

df_sem_outliers.describe()

df_sem_outliers.info()

"""# Divisão de treino e teste a ser usado no treinamento e avaliação"""

X = df_sem_outliers.drop('quantidade', axis=1)
y = df_sem_outliers['quantidade']
#
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

"""# Treinamento e avaliação WMAPE sem outliers"""

"""# RandomForestRegressor"""



# -----------------------------
# Métrica WMAPE
# -----------------------------
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

wmape_scorer = make_scorer(wmape, greater_is_better=False)  # menor = melhor

# -----------------------------
# Feature Engineer
# -----------------------------
class StatsFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stats_pdv_mean = None
        self.stats_prod_mean = None
        self.stats_semana_mean = None
        self.stats_pdv_median = None
        self.stats_prod_median = None
        self.stats_semana_median = None

    def fit(self, X, y=None):
        df = X.copy()
        df["quantidade"] = y

        # Médias
        self.stats_pdv_mean = df.groupby("pdv")["quantidade"].mean().rename("pdv_mean_qtd")
        self.stats_prod_mean = df.groupby("produto")["quantidade"].mean().rename("prod_mean_qtd")
        self.stats_semana_mean = df.groupby("semana")["quantidade"].mean().rename("semana_mean_qtd")

        # Medianas
        self.stats_pdv_median = df.groupby("pdv")["quantidade"].median().rename("pdv_median_qtd")
        self.stats_prod_median = df.groupby("produto")["quantidade"].median().rename("prod_median_qtd")
        self.stats_semana_median = df.groupby("semana")["quantidade"].median().rename("semana_median_qtd")

        return self

    def transform(self, X):
        df = X.copy()

        df = df.join(self.stats_pdv_mean, on="pdv")
        df = df.join(self.stats_prod_mean, on="produto")
        df = df.join(self.stats_semana_mean, on="semana")

        df = df.join(self.stats_pdv_median, on="pdv")
        df = df.join(self.stats_prod_median, on="produto")
        df = df.join(self.stats_semana_median, on="semana")

        return df.fillna(0)

# -----------------------------
# Pipeline
# -----------------------------
categorical_features = ["pdv", "produto"]
numerical_features = ["semana", "pdv_mean_qtd", "prod_mean_qtd", "semana_mean_qtd",
                      "pdv_median_qtd", "prod_median_qtd", "semana_median_qtd"]

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", StandardScaler())
])

col_transformer = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, numerical_features)
    ]
)


pipeline = Pipeline(steps=[
    ("feature_engineer", StatsFeatureEngineer()),
    ("preprocessor", col_transformer),
    ("regressor", RandomForestRegressor(random_state=42))
])

# -----------------------------
# Cross-validation simples
# -----------------------------
scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=wmape_scorer)
print("WMAPE médio (cross-val):", -scores.mean())
print("WMAPE por fold:", -scores)

# -----------------------------
# GridSearchCV
# -----------------------------
param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [10, 20, None],
    "regressor__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring=wmape_scorer,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Melhores hiperparâmetros:", grid_search.best_params_)
print("Melhor WMAPE (cv):", -grid_search.best_score_)

# -----------------------------
# Avaliação final no holdout
# -----------------------------
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)
print("WMAPE test:", wmape(y_test, y_pred))

# -----------------------------
# Submissão
# -----------------------------
# Em df_teste colunas pdv e  produto devem ser object.
df_teste['pdv'] = df_teste['pdv'].astype(str)
df_teste['produto'] = df_teste['produto'].astype(str)

y_submit = best_pipeline.predict(df_teste)
df_submit = df_teste[["semana", "pdv", "produto"]].copy()
df_submit["quantidade"] = np.round(y_submit).astype(int)
df_submit.to_csv("submission.csv", sep=";", index=False, encoding="utf-8")
print("Arquivo submission.csv gerado!")

"""# Pipeline final"""
"""

# -----------------------------
# Métrica WMAPE
# -----------------------------
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

wmape_scorer = make_scorer(wmape, greater_is_better=False)

# -----------------------------
# Feature Engineer
# -----------------------------
class StatsFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = X.copy()
        df["quantidade"] = y

        # Médias
        self.stats_pdv_mean = df.groupby("pdv")["quantidade"].mean().rename("pdv_mean_qtd")
        self.stats_prod_mean = df.groupby("produto")["quantidade"].mean().rename("prod_mean_qtd")
        self.stats_semana_mean = df.groupby("semana")["quantidade"].mean().rename("semana_mean_qtd")

        # Medianas
        self.stats_pdv_median = df.groupby("pdv")["quantidade"].median().rename("pdv_median_qtd")
        self.stats_prod_median = df.groupby("produto")["quantidade"].median().rename("prod_median_qtd")
        self.stats_semana_median = df.groupby("semana")["quantidade"].median().rename("semana_median_qtd")

        return self

    def transform(self, X):
        df = X.copy()
        df = df.join(self.stats_pdv_mean, on="pdv")
        df = df.join(self.stats_prod_mean, on="produto")
        df = df.join(self.stats_semana_mean, on="semana")
        df = df.join(self.stats_pdv_median, on="pdv")
        df = df.join(self.stats_prod_median, on="produto")
        df = df.join(self.stats_semana_median, on="semana")
        return df.fillna(0)

# -----------------------------
# Pipeline
# -----------------------------
# Colunas
categorical_features = ["pdv", "produto"]
numerical_features = ["semana", "pdv_mean_qtd", "prod_mean_qtd", "semana_mean_qtd",
                      "pdv_median_qtd", "prod_median_qtd", "semana_median_qtd"]

# Transformadores
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", StandardScaler())
])

col_transformer = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, numerical_features)
    ]
)

# Pipeline completo
pipeline = Pipeline(steps=[
    ("feature_engineer", StatsFeatureEngineer()),
    ("preprocessor", col_transformer),
    ("regressor", RandomForestRegressor(random_state=42))
])

# -----------------------------
# Cross-validation
# -----------------------------
scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=wmape_scorer)
print("WMAPE médio (cross-val):", -scores.mean())
print("WMAPE por fold:", -scores)

# -----------------------------
# GridSearchCV
# -----------------------------
param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [10, 20, None],
    "regressor__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring=wmape_scorer,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
print("Melhores hiperparâmetros:", grid_search.best_params_)
print("Melhor WMAPE (cv):", -grid_search.best_score_)

# -----------------------------
# Avaliação final
# -----------------------------
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)
print("WMAPE test:", wmape(y_test, y_pred))

# -----------------------------
# Submissão
# -----------------------------
# Garante que pdv e produto do dataset de teste sejam strings para o pipeline
df_teste['pdv'] = df_teste['pdv'].astype(str)
df_teste['produto'] = df_teste['produto'].astype(str)

# Predição
y_submit = best_pipeline.predict(df_teste)

# Monta dataframe de submissão
df_submit = df_teste[["semana", "pdv", "produto"]].copy()
df_submit["quantidade"] = np.round(y_submit).astype(int)

# Converte todas colunas para inteiro antes de salvar
df_submit["semana"] = df_submit["semana"].astype(int)
df_submit["pdv"] = df_submit["pdv"].astype(int)
df_submit["produto"] = df_submit["produto"].astype(int)

# Salva CSV final para submissão
df_submit.to_csv("submission.csv", sep=";", index=False, encoding="utf-8")
print("Arquivo submission.csv gerado!")
"""

