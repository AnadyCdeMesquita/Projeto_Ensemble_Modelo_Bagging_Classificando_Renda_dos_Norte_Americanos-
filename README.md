# Projeto_Ensemble_Modelo_Bagging_Classificando_Renda_dos_Norte_Americanos-
Projeto desenvolvido como atividade prática (Hands-On) da Preditiva Analytics. O objetivo é prever se um indivíduo possui renda superior ou inferior a US$ 50 mil/ano a partir de dados de um censo americano, comparando modelos individuais com modelos de ensemble baseados em Bagging.

🎯 Problema
Tipo: classificação binária
Variável alvo (target): renda >50K (1) ou <=50K (0)
Base: censo_limpo.csv — 30.139 registros já tratados
Desbalanceamento: aproximadamente 25% da base pertence à classe >50K
🧾 Variáveis
Tipo	Variáveis
Numéricas	idade, anos_estudo, ganho_capital, perda_capital, horas_por_semana
Categóricas	classe_trabalho, escolaridade, estado_civil, ocupacao, relacionamento, raca, sexo, pais_natal

As colunas peso_final e renda foram removidas antes da modelagem.

🛠️ Metodologia
Análise exploratória: pairplot das variáveis numéricas segmentado pela variável alvo (pairplot_alvo_numericas.png).
Pré-processamento:
One-hot encoding das categóricas com pd.get_dummies(drop_first=True) → 96 features
Padronização das numéricas com StandardScaler (ajustado apenas no treino, evitando data leakage)
Divisão dos dados: 80% treino / 20% teste, estratificada pela variável alvo (random_state=42).
Modelos comparados:
Random Forest (100 árvores)
Bagging com Regressão Logística (100 estimadores)
Árvore de Decisão (individual)
Regressão Logística (individual)
Avaliação: acurácia, precisão, recall, ROC AUC e matriz de confusão, em treino e teste, para diagnosticar overfitting.
📊 Resultados (conjunto de teste)
Modelo	Acurácia	ROC AUC	Precisão	Recall	Δ Acurácia (treino − teste)
Random Forest	0,8464	0,8932	0,7167	0,6336	0,1307
Bagging + Regressão Logística	0,8461	0,9007	0,7320	0,6023	0,0026
Regressão Logística	0,8461	0,9008	0,7320	0,6023	0,0029
Árvore de Decisão	0,8185	0,7742	0,6436	0,6076	0,1586
🔎 Principais aprendizados
Modelos lineares são estáveis: a Regressão Logística (isolada ou com Bagging) apresentou diferença mínima entre treino e teste, sem sinais de overfitting, e obteve o melhor ROC AUC (~0,90).
Bagging não melhorou a Regressão Logística: por ser um modelo de baixa variância, ela se beneficia pouco da reamostragem. O Bagging é mais eficaz com aprendizes instáveis, como árvores.
Bagging de árvores (Random Forest) foi muito superior à árvore individual: ROC AUC de 0,774 para 0,893 e acurácia de 0,819 para 0,846.
Árvores sem poda sofrem overfitting: a Random Forest atingiu 97,7% de acurácia no treino contra 84,6% no teste.
Recall da classe >50K é o ponto fraco: ~60–63% em todos os modelos, reflexo do desbalanceamento da base. Em contrapartida, a Random Forest foi a que mais identificou corretamente quem ganha mais de 50K.
🚀 Próximos passos
Ajustar hiperparâmetros (max_depth, min_samples_leaf, max_features) para controlar o overfitting da Random Forest
Tratar o desbalanceamento (class_weight, SMOTE) e ajustar o threshold de decisão para elevar o recall
Testar Boosting (Gradient Boosting, XGBoost, LightGBM)
Validação cruzada e análise de importância das variáveis
