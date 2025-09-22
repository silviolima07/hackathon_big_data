# Previsão de Vendas - Hackathon Big Data

Este projeto tem como objetivo prever a quantidade de vendas de produtos em pontos de venda (PDVs) para determinadas semanas, utilizando dados reais de transações. O código principal está disponível em um notebook Jupyter/Colab.

## Como Executar o Notebook

1. **Abra o notebook no Google Colab ou Jupyter Notebook.**
2. **Execute cada célula sequencialmente, desde o início até o final, sem pular nenhuma etapa.**
    - É importante rodar todas as células para garantir o correto carregamento das bibliotecas, leitura do dataset e criação das features.
3. **Configuração do caminho do dataset:**  
    - Antes de rodar a célula de leitura dos dados, **configure o caminho do arquivo do dataset** para o local onde ele está disponível.
    - Neste projeto, o dataset foi obtido a partir do **Google Drive**. Certifique-se de montar o Google Drive (no Colab: `from google.colab import drive; drive.mount('/content/drive')`) e ajustar o caminho do arquivo Parquet conforme o local em que está salvo no seu Drive.
    - Exemplo de caminho:  
      `'/content/drive/MyDrive/hackathon_big_data/dataset.parquet'`

4. Após executar todas as células, o arquivo de submissão será gerado como `submission.csv`.

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
- Foram feitos testes com apenas 10% dos dados como demonstração.

### 4. **Engenharia de Features**
- São criadas variáveis de lag (valores de vendas de semanas anteriores) e médias móveis para cada combinação de PDV e produto.
- Missing values nas features de lag/médias móveis são preenchidos com zero.

### 5. **Criação do Dataset de Teste**
- Um conjunto de teste é gerado simulando previsões para as semanas 1 a 5 de janeiro, usando combinações distintas de PDV e produto.

### 6. **Preparação para Submissão**
- O dataset de submissão recebe as mesmas features criadas para o treino, extraídas dos dados históricos.
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
- scikit-learn

### Instalação das dependências

No Colab, elas já estão disponíveis. Caso rode localmente, instale com:

```bash
pip install pandas numpy pyarrow scikit-learn
```

## Observações

- O pipeline lida com dados faltantes e outliers para garantir robustez.
- O modelo pode ser ajustado para utilizar mais dados ou outras técnicas de Machine Learning.
- O código está preparado para ser adaptado a diferentes datasets e desafios de previsão de vendas.

## Autor

Desenvolvido por [silviolima07](https://github.com/silviolima07) para o Hackathon Big Data.
