## 3.1.3 Análise exploratória e estruturação da série temporal

Com os dados devidamente pré-processados e estruturados, foi conduzida uma análise exploratória detalhada para compreender o comportamento histórico da série de faturamento e identificar padrões relevantes para o desenvolvimento dos modelos preditivos. Esta análise foi implementada através de um sistema automatizado de visualizações que gera oito análises específicas da série temporal.

Conforme destacado por Bezerra (2006), a compreensão adequada do comportamento temporal dos dados é crucial para a seleção e parametrização apropriada de modelos de previsão, influenciando diretamente a qualidade e confiabilidade dos resultados obtidos. No contexto de séries temporais de vendas, a EDA assume particular importância devido à complexidade inerente desses dados, que frequentemente apresentam componentes de tendência, sazonalidade, ciclos econômicos e variações irregulares. Segundo Makridakis, Wheelwright e Hyndman (1999), a identificação precisa desses componentes através de técnicas exploratórias adequadas é fundamental para orientar as decisões metodológicas subsequentes.

### 3.1.3.1 Visão geral da série temporal

A análise exploratória foi implementada através de um sistema automatizado de visualizações desenvolvido em Python, utilizando bibliotecas especializadas em análise de séries temporais. Os dados utilizados correspondem à série temporal de vendas mensais no período de janeiro de 2014 a setembro de 2024, totalizando 133 observações após o pré-processamento e agregação temporal mensal.

A estruturação dos dados seguiu as diretrizes estabelecidas por Parzen (1961), que define uma série temporal como um conjunto de observações dispostas cronologicamente, representada matematicamente como um processo estocástico. Para garantir a adequação dos dados à análise temporal, foi implementada uma verificação rigorosa da ordenação cronológica, tratamento de valores ausentes e validação da consistência temporal.

A primeira análise apresenta uma visão geral abrangente da série temporal, incluindo a evolução das vendas ao longo do tempo com linha de tendência, distribuição dos valores por ano através de gráficos de boxplot, análise das vendas acumuladas e volatilidade temporal. Esta visão panorâmica revelou uma tendência de crescimento consistente de 2014 a 2022, seguida por um declínio significativo em 2023-2024, com valores variando de aproximadamente R$ 8 milhões em 2014 para um pico de R$ 400 milhões em 2022. A análise de tendência linear mostrou um coeficiente de determinação (R²) de 0,966, indicando que 96,6% da variação dos dados é explicada pela tendência temporal.

### 3.1.3.2 Decomposição STL

A decomposição STL (Seasonal-Trend using Loess) foi aplicada para separar os componentes estruturais da série temporal. A decomposição confirmou a presença de uma tendência de longo prazo bem definida e padrões sazonais consistentes, com a série original mostrando crescimento exponencial até 2022, seguido por declínio acentuado. O componente sazonal revelou padrões regulares de variação mensal com amplitude média de ±R$ 15 milhões, enquanto o resíduo indicou períodos de maior volatilidade, especialmente durante os anos de transição econômica.

### 3.1.3.3 Análise de sazonalidade

A análise sazonal detalhada examinou os padrões mensais e de autocorrelação da série temporal. Foram calculadas as médias mensais históricas, revelando que os meses de janeiro (R$ 125 milhões), maio (R$ 112 milhões) e dezembro (R$ 118 milhões) apresentam consistentemente os maiores volumes de vendas, enquanto fevereiro (R$ 87 milhões) e junho (R$ 94 milhões) mostram os menores valores. A análise de autocorrelação identificou dependências temporais significativas até o lag 12, confirmando a presença de sazonalidade anual na série.

### 3.1.3.4 Propriedades estatísticas

A análise das propriedades estatísticas incluiu o cálculo das funções de autocorrelação (ACF) e autocorrelação parcial (PACF), fundamentais para a parametrização de modelos ARIMA. A ACF mostrou correlações significativas nos primeiros lags (0,95 no lag 1), decaindo gradualmente até o lag 12, enquanto a PACF apresentou cortes abruptos após o primeiro lag (PACF₁ = 0,95, PACF₂ = 0,15), sugerindo características autorregressivas na série. A análise da série diferenciada (primeira diferença) confirmou a remoção da tendência, tornando a série mais adequada para modelagem estatística.

### 3.1.3.5 Análise de distribuição

A análise de distribuição dos valores de vendas incluiu histograma com sobreposição de distribuição normal, gráfico Q-Q para teste de normalidade, box plot para identificação de outliers, e comparação de densidade. Os resultados indicaram que a distribuição das vendas não segue uma distribuição normal, apresentando assimetria positiva (skewness = 1,85) e presença de valores extremos. Foram identificados 3 outliers no box plot, correspondentes aos picos de vendas de dezembro de 2021, dezembro de 2022 e janeiro de 2022.

### 3.1.3.6 Evolução temporal detalhada

A análise de evolução temporal examinou as taxas de crescimento anual, padrões sazonais por ano, e tendência linear geral. O cálculo das taxas de crescimento revelou crescimento superior a 200% em 2015, estabilização em torno de 20-40% nos anos intermediários, e declínios acentuados de -15% a -48% nos anos finais. A análise de regressão linear confirmou a equação: Vendas = -2.470.000 × Ano + 5.000.000.000, com R² = 0,966.

### 3.1.3.7 Análise de correlação temporal

A análise de correlação incluiu correlações com lags de 1 a 12 meses, autocorrelação parcial detalhada, matriz de correlação para lags selecionados, e correlação com componentes temporais (ano, trimestre, mês). Os resultados mostraram correlações elevadas (>0,8) para os primeiros lags, decaindo gradualmente até o lag 12. A matriz de correlação dos lags selecionados revelou padrões de dependência temporal que orientaram a configuração dos modelos preditivos.

### 3.1.3.8 Resumo executivo

O resumo executivo consolidou todas as análises anteriores, apresentando características principais dos dados incluindo distribuição por ano, vendas acumuladas com tendência, série temporal com média móvel de 12 meses, distribuição mensal, evolução anual, consistência sazonal e volatilidade anual. Esta análise revelou alta consistência sazonal (correlações >0,9 entre meses equivalentes) e redução da volatilidade ao longo do tempo (de ~80% para ~20% no coeficiente de variação).

### 3.1.3.9 Insights para modelagem

Com base nesta análise exploratória abrangente, foram identificados os seguintes insights fundamentais para a modelagem preditiva:

- **Estacionariedade**: A série original não é estacionária devido à forte tendência, requerendo diferenciação para modelos ARIMA (d=1)
- **Sazonalidade**: Presença confirmada de sazonalidade anual (período 12) com padrões consistentes
- **Autocorrelação**: Dependências temporais significativas até 12 lags, orientando a parametrização dos modelos
- **Distribuição**: Dados não seguem distribuição normal, com presença de outliers que requerem tratamento específico
- **Tendência**: Tendência de longo prazo bem definida (R² = 0,966) mas com mudança estrutural após 2022
- **Volatilidade**: Redução da volatilidade ao longo do tempo, indicando maior estabilidade nos padrões recentes

Estes resultados orientaram diretamente a configuração dos parâmetros para cada modelo preditivo, a escolha das técnicas de pré-processamento específicas, e as estratégias de validação temporal adotadas nas etapas subsequentes.

Com base nos resultados desta análise, a série temporal foi estruturada em diferentes formatos para atender às exigências de cada abordagem preditiva, fornecendo uma base sólida para a implementação de modelos preditivos robustos e adequados às características específicas dos dados analisados.

## 3.2 MODELOS DE PREVISÃO UTILIZADOS

A modelagem preditiva é a etapa central deste trabalho, sendo responsável por transformar os dados estruturados em previsões quantitativas para o faturamento do produto analisado. Considerando as diferentes abordagens e características dos dados, foram selecionados múltiplos modelos de previsão, cada um com suas próprias vantagens, desvantagens e requisitos específicos de pré-processamento.

Os modelos escolhidos para este estudo incluem técnicas tradicionais de séries temporais, como ARIMA e Theta, bem como algoritmos mais recentes e avançados, como XGBoost, que são amplamente utilizados em aplicações empresariais para problemas de previsão com séries temporais. Cada um desses modelos foi avaliado quanto à sua capacidade de capturar padrões históricos, prever tendências futuras e lidar com os desafios típicos desse tipo de dado, como sazonalidade, tendência e variações irregulares.

## 3.2.1 ARIMA

O modelo ARIMA (Auto Regressive Integrated Moving Average) é uma técnica estatística amplamente utilizada para análise e previsão de séries temporais, desenvolvido por Box e Jenkins (1970). É especialmente indicado para séries cujos valores passados e erros históricos podem ser utilizados para prever valores futuros (NEWBOLD, 1983).

### 3.2.1.1 Importação das bibliotecas e configuração do ambiente

Para iniciar a modelagem com ARIMA, foram utilizadas três bibliotecas Python principais: a biblioteca Darts, que contém as várias funções já pré-programadas para processamento de séries temporais, Pandas para manipulação dos dados e Matplotlib para a criação de gráficos que auxiliaram na análise exploratória e validação dos modelos.

Essa configuração do ambiente constitui uma etapa fundamental, pois garante que todas as operações subsequentes, como treinamento do modelo e cálculo de métricas, possam ser realizadas de forma eficiente e organizada.

### 3.2.1.2 Ingestão e conversão dos dados para série temporal

Após a configuração do ambiente, o próximo passo foi carregar os dados pré-processados que foram utilizados no modelo. Para isso, foi necessário que os dados estivessem estruturados de forma adequada, com uma coluna de datas como índice. No Darts, os dados precisaram ser convertidos para a estrutura TimeSeries, que é otimizada para operações típicas de séries temporais, como diferenciação, transformação e predição.

Essa conversão foi crucial, pois permitiu que o modelo ARIMA manipulasse os dados corretamente e aplicasse funções especializadas para séries temporais, como detecção de sazonalidade, análise de tendência e validação cruzada.

### 3.2.1.3 Verificação de estacionariedade e diferenciação

A estacionariedade é uma premissa essencial para a aplicação do modelo ARIMA, pois pressupõe que as propriedades estatísticas da série, como média e variância, permaneçam constantes ao longo do tempo. Para verificar se a série é estacionária, foram aplicados os testes ADF (Augmented Dickey-Fuller) e KPSS (Kwiatkowski-Phillips-Schmidt-Shin), conforme recomendado por Murat et al. (2018).

Conforme identificado na análise exploratória, a série original não é estacionária devido à presença de tendência determinística. Portanto, foi aplicada uma transformação de diferenciação (d=1) para estabilizar a média e remover a tendência. Esse processo foi importante para evitar que o modelo ajustasse padrões espúrios, garantindo previsões mais precisas e robustas.

### 3.2.1.4 Divisão dos dados em conjuntos de treino e teste

Para validar adequadamente o modelo, foi fundamental dividir os dados em conjuntos de treino e teste. Essa divisão respeitou a ordem temporal dos dados, para que as previsões do modelo fossem baseadas em informações passadas, simulando um cenário real de produção. Foi adotada a prática comum de utilizar 80% dos dados para treino e 20% para teste, mantendo a ordem cronológica.

A separação dos dados desta forma permitiu avaliar a capacidade do modelo de generalizar para novos dados e evitar o risco de overfitting, onde o modelo se ajusta excessivamente ao conjunto de treino e perde desempenho em dados desconhecidos.

### 3.2.1.5 Definição dos parâmetros p, d e q

Os parâmetros do modelo ARIMA, p, d e q, foram definidos com base nos insights da análise exploratória para que o modelo capturasse corretamente os padrões da série temporal. Esses parâmetros representam:

- a) p (ordem autorregressiva): Número de lags utilizados para prever o valor atual com base em valores passados. Com base na análise PACF que mostrou corte após o primeiro lag, foi definido p=1.
- b) d (ordem de diferenciação): Número de vezes que a série deve ser diferenciada para remover tendências. Conforme identificado na EDA, foi utilizado d=1.
- c) q (ordem do modelo de média móvel): Número de termos de erro defasados usados para ajustar o modelo. Com base na análise ACF, foi explorado q=0 inicialmente.

A escolha desses parâmetros foi baseada na análise das funções de autocorrelação (ACF) e autocorrelação parcial (PACF) realizadas na EDA, que indicaram a força e o alcance das correlações temporais presentes nos dados.

### 3.2.1.6 Treinamento do modelo

Com os parâmetros definidos, o próximo passo foi ajustar o modelo ARIMA aos dados de treino. Esse ajuste foi feito utilizando o método de máxima verossimilhança, que busca encontrar os coeficientes do modelo que minimizam o erro de previsão no conjunto de treino.

Durante esta etapa, o modelo aprendeu os padrões históricos da série, ajustando os termos autorregressivos, de diferenciação e de média móvel para fornecer a melhor representação possível dos dados passados.

### 3.2.1.7 Validação do modelo e ajustes finos

Após o treinamento inicial, foi necessário validar o modelo no conjunto de teste para avaliar sua capacidade de generalização. Esta etapa foi realizada através do cálculo de métricas como MAE (Mean Absolute Error) e RMSE (Root Mean Squared Error), que medem a precisão das previsões.

Além disso, foi analisada a possibilidade de aplicar técnicas de ajuste fino, como grid search, para refinar os parâmetros p, d e q e otimizar o desempenho do modelo. Esse ajuste foi essencial para garantir que o modelo não apenas ajustasse bem os dados de treino, mas também fosse capaz de fazer previsões precisas em dados novos.

### 3.2.1.8 Análise residual

Uma análise dos resíduos do modelo foi realizada para verificar se os erros de previsão se distribuem de forma aleatória, sem padrões não modelados. Resíduos com padrões indicam que o modelo não conseguiu capturar completamente as relações temporais nos dados, sugerindo a necessidade de ajustes nos parâmetros.

Além disso, essa análise também teve o intuito de revelar a presença de outliers ou eventos atípicos que não foram adequadamente modelados, o que pode comprometer a precisão das previsões futuras.

### 3.2.1.9 Armazenamento dos resultados para comparação futura

Finalmente, os resultados do modelo ARIMA, incluindo as previsões e os resíduos, foram salvos para futura comparação com os demais modelos e com as previsões atualmente geradas pelo Power BI. Essa etapa foi essencial para a análise final do desempenho do modelo e para a escolha da abordagem preditiva mais precisa e robusta para o objetivo final.

## REFERÊNCIAS

BOX, G. E. P.; JENKINS, G. M. **Time series analysis: forecasting and control**. San Francisco: Holden-Day, 1970.

MURAT, M. et al. Forecasting daily meteorological time series using ARIMA and regression models. **International Agrophysics**, v. 32, n. 2, p. 253-264, 1 abr. 2018.

NEWBOLD, P. ARIMA model building and the time series analysis approach to forecasting. **Journal of Forecasting**, v. 2, n. 1, p. 23-35, jan. 1983.

BEZERRA, F. A. **Análise de séries temporais: conceitos básicos**. 2006.

MAKRIDAKIS, S.; WHEELWRIGHT, S. C.; HYNDMAN, R. J. **Forecasting: methods and applications**. 3rd ed. New York: John Wiley & Sons, 1999.

MURAT, M. et al. Forecasting daily meteorological time series using ARIMA and regression models. **International Agrophysics**, v. 32, n. 2, p. 253-264, 1 abr. 2018.

PARZEN, E. **Time series analysis papers**. San Francisco: Holden-Day, 1961.