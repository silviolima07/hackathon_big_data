# -*- coding: utf-8 -*-
"""

# Colunas do dataset de teste:

 - semana (número inteiro): número da semana (1 a 5 de janeiro/2023)
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

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import warnings
warnings.filterwarnings("ignore")



"""# Modelo deve prever a quantidade com estas colunas no dataset de teste, gerando a coluna quantidade"""

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

df_final = df_sem_outliers.copy()
print("\nDataset final\n",df_final)

 # Categoricos
le_pdv = LabelEncoder() # Separate encoder for pdv
le_produto = LabelEncoder() # Separate encoder for produto

df_final['pdv'] = le_pdv.fit_transform(df_sem_outliers['pdv'])
df_final['produto'] = le_produto.fit_transform(df_sem_outliers['produto'])

"""# Divisão de treino e teste a ser usado no treinamento e avaliação"""

X = df_final.drop('quantidade', axis=1)
y = df_final['quantidade']
#
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

"""# Treinamento e avaliação WMAPE sem outliers"""

"""# XGBRegressor"""



# -----------------------------
# Métrica WMAPE
# -----------------------------
def wmape(y_pred, y_true):
    y_true = y_true.get_label() # Extract labels from DMatrix
    return 'WMAPE', np.sum(np.abs(np.expm1(y_true) - np.expm1(y_pred))) / np.sum(np.abs(np.expm1(y_true))) # Modified WMAPE to use inverse transformed values


# Modelo
param = {
    'objective': 'reg:squarederror', # Changed objective as reg:linear is deprecated
    "booster" : "gbtree",
    'eta': 0.03,
    'max_depth':10,
    'subsample':0.9,
    'colsample_bytree':0.7
}

# Apply log1p transformation to the target variable y
y_train_transformed = np.log1p(y_train)
y_test_transformed = np.log1p(y_test)

dtrain = xgb.DMatrix(X_train, y_train_transformed) # Use transformed y_train
dvalid = xgb.DMatrix(X_test, y_test_transformed) # Use transformed y_test

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

print('Treinamento')
print(dtrain.get_data())
print(dtrain.get_label())


# Treinamento
print("Treinando em 100 epochs")
gbm = xgb.train(
            param,
            dtrain,
            num_boost_round=100,
            evals=watchlist,
            early_stopping_rounds=100,
            custom_metric=wmape,
            verbose_eval=100
)
# Predict treino
yhat = gbm.predict(xgb.DMatrix(X_test))
# Modify wmape to handle both DMatrix and pandas Series

# This wmape is for evaluation outside of xgb.train
def wmape_eval(y_true, y_pred):
    # Inverse transform y_true and y_pred for WMAPE calculation
    y_true_original = np.expm1(y_true)
    y_pred_original = np.expm1(y_pred)
    return np.sum(np.abs(y_true_original - y_pred_original)) / np.sum(np.abs(y_true_original))


# Use the evaluation version of wmape here
# Pass the transformed y_test and yhat to the evaluation wmape, it will inverse transform internally
print("WMAPE:", wmape_eval(y_test_transformed, yhat))


# -----------------------------
# Submissão
# -----------------------------

"""
### Preparing Actual Test Data for Prediction

To make predictions, your test data must have the same columns and be preprocessed
in the same way as your training data (X_train).

Specifically, the 'pdv' and 'produto' columns in your test data must contain
the **original string IDs** that were used to fit the LabelEncoder during training.
Then, you must use the *same* fitted LabelEncoder to transform these string IDs
into numerical labels before feeding them to the model.

Replace the line below with code to load your actual test dataset.
"""
try:
    # Prepare a sample df_teste with original string IDs from training data

    # Get unique original pdv and produto IDs from the training data before encoding
    # We can access these from df_sem_outliers before the LabelEncoder was applied
    # Assuming df_sem_outliers still contains the original string IDs

    if 'pdv' in df_sem_outliers.columns and 'produto' in df_sem_outliers.columns and df_sem_outliers['pdv'].dtype == 'object':
        unique_pdv_ids = df_sem_outliers['pdv'].unique()
        unique_produto_ids = df_sem_outliers['produto'].unique()

        # Select a few sample IDs for df_teste
        sample_pdv_ids = unique_pdv_ids[:5] # Take the first 3 unique pdv IDs
        sample_produto_ids = unique_produto_ids[:5] # Take the first 3 unique produto IDs

        # Create the sample df_teste DataFrame
        data = {
        'semana': [1, 2, 3, 4, 5],
        'pdv':     [sample_pdv_ids[0], sample_pdv_ids[1], sample_pdv_ids[2], sample_pdv_ids[3], sample_pdv_ids[4]], # Use sample original IDs
        'produto': [sample_produto_ids[0], sample_produto_ids[1], sample_produto_ids[2], sample_produto_ids[3], sample_produto_ids[4]] # Use sample original IDs
        }
        df_teste_correct_format = pd.DataFrame(data)
        df_teste_correct_format.to_csv('df_teste_correct_format.csv', index=False)

        df_actual_test = pd.read_csv('/content/df_teste_correct_format.csv')

        print("Dataset de submissao:\n", df_actual_test.head())

        # Create mapping dictionaries from the fitted LabelEncoder classes
        # Use the separate encoders for pdv and produto
        pdv_mapping = {label: idx for idx, label in enumerate(le_pdv.classes_)}
        produto_mapping = {label: idx for idx, label in enumerate(le_produto.classes_)}

        # Apply mapping, assigning a default value (-1) to unseen labels
        # Using -1 as a placeholder for unseen values. You might need to adjust
        # this based on how your model handles unseen categorical values.
        df_actual_test['pdv_encoded'] = df_actual_test['pdv'].map(pdv_mapping).fillna(-1).astype(int)
        df_actual_test['produto_encoded'] = df_actual_test['produto'].map(produto_mapping).fillna(-1).astype(int)

        # Select the features used for training (using the new encoded columns)
        features_for_prediction = ['pdv_encoded', 'produto_encoded', 'semana'] # Use the new encoded column names
        df_actual_test_processed = df_actual_test[features_for_prediction].copy()

        # Rename columns to match the training features if necessary (e.g., if X_train had 'pdv' and 'produto')
        df_actual_test_processed.rename(columns={'pdv_encoded': 'pdv', 'produto_encoded': 'produto'}, inplace=True)


        # Create DMatrix for prediction
        dtest_actual = xgb.DMatrix(df_actual_test_processed)

        # Make predictions using the trained model
        print("\nMaking predictions...")
        test_probs_actual = gbm.predict(dtest_actual)

        # Apply the best weight (from previous analysis) and inverse transform
        peso_final = 0.999 # Using the best weight found in cell xgJctDhVF716
        predictions_actual = np.expm1(test_probs_actual * peso_final)

        # Add predictions to the original test DataFrame or a copy
        # Add to the DataFrame used for prediction before renaming for submission
        df_actual_test_processed['quantidade'] = predictions_actual

        # Format for submission (adjust column names and types as needed)
        # Assuming the submission requires the original 'pdv' and 'produto' IDs
        # If submission requires encoded IDs, use df_actual_test_processed directly
        submissao = pd.DataFrame({
              "semana": df_actual_test['semana'].astype(int), # Use original 'semana' if needed
              "pdv": df_actual_test['pdv'].astype(int), # Use original 'pdv' ID
              "produto": df_actual_test['produto'].astype(int), # Use original 'produto' ID
              "quantidade": df_actual_test_processed['quantidade'].astype(int) # Use predicted quantity
          })

        # If you had an 'Id' column and need a submission file:
        # submission_actual = pd.DataFrame({"Id": test_ids, "quantidade": predictions_actual})
        # submission_actual.to_csv("actual_predictions_submission.csv", index=False)
        # print("\nSubmission file 'actual_predictions_submission.csv' created.")


        print("Submissao:\n", submissao.head())
        print(submissao.info())
        submissao.to_csv("submission.csv", index=False, sep=';', encoding="utf-8") # Changed sep to ; and added encoding
        print("\nArquivo submission.csv gerado!")

    else:
        print("Could not create sample df_teste with original IDs. 'df_sem_outliers' might not contain the original 'pdv' and 'produto' string IDs.")
        print("Please ensure 'df_sem_outliers' or a similar DataFrame containing the original IDs is available.")

except FileNotFoundError:
    print("Error: Test data file not found. Please update the file path.")
except ValueError as e:
    print(f"Error during preprocessing (LabelEncoding): {e}")
    print("This error might still occur if the mapping process failed unexpectedly.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
