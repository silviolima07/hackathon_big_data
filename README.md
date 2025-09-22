# Previsão de Vendas - Hackathon Big Data

Este projeto tem como objetivo prever a quantidade de vendas de produtos em pontos de venda (PDVs) para determinadas semanas, utilizando dados reais de transações. O código principal está no arquivo `hacka_vendas.py`, desenvolvido durante um hackathon de Big Data.

## Descrição do Processo

O código realiza as seguintes etapas:

### 1. **Leitura dos Dados**
- Os dados são lidos a partir de arquivos Parquet usando o PyArrow.
- Os arquivos são concatenados para formar um único DataFrame pandas.

### 2. **Exploração e Limpeza**
- Estatísticas descritivas e verificação de valores ausentes são feitas para entender a qualidade do dataset.
- Colunas com alto percentual de valores ausentes (`pdv`, `produto`) são substituídas por identificadores internos.
- Linhas com valores ausentes em `quantity` são removidas.
- Outliers (valores negativos ou fora do intervalo interquartil) são removidos.

### 3. **Preparação dos Dados**
- As colunas são renomeadas para facilitar o processamento.
- É extraído o número da semana do ano a partir da data de transação.
- Apenas 10% dos dados são utilizados para treinamento como demonstração.

### 4. **Engenharia de Features**
- São criadas variáveis de lag (valores de vendas de semanas anteriores) e médias móveis para cada combinação de PDV e produto.
- Missing values nas features de lag/médias móveis são preenchidos com zero.

### 5. **Criação do Dataset de Teste**
- Um conjunto de teste é gerado simulando previsões para as semanas 1 a 5 de janeiro, usando combinações distintas de PDV e produto.

### 6. **Preparação para Submissão**
- O dataset de teste recebe as mesmas features criadas para o treino, extraídas dos dados históricos.
- Garante-se que todas as features esperadas pelo modelo estejam presentes.

### 7. **Treinamento e Avaliação do Modelo**
- Utiliza-se um modelo de regressão Random Forest para prever a quantidade de vendas.
- O modelo é avaliado usando o WMAPE (Weighted Mean Absolute Percentage Error).

### 8. **Geração das Previsões**
- O modelo é aplicado ao dataset de submissão, gerando as previsões de vendas.
- As previsões são arredondadas e formatadas para envio (como CSV).

## Principais Dependências

- pandas
- numpy
- pyarrow
- duckdb
- scikit-learn
- seaborn
- matplotlib

## Como Executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
   Ou individualmente:
   ```bash
   pip install pandas numpy pyarrow duckdb scikit-learn seaborn matplotlib
   ```

2. Execute o script principal:
   ```bash
   python hacka_vendas.py
   ```

3. O arquivo de submissão será gerado como `submission.csv`.

## Observações

- O pipeline lida com dados faltantes e outliers para garantir robustez.
- O modelo pode ser ajustado para utilizar mais dados ou outras técnicas de Machine Learning.
- O código está preparado para ser adaptado a diferentes datasets e desafios de previsão de vendas.

## Autor

Desenvolvido por [silviolima07](https://github.com/silviolima07) para o Hackathon Big Data.

