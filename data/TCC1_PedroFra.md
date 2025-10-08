## UNIVERSIDADE DO VALE DO RIO DOS SINOS (UNISINOS) UNIDADE ACADÊMICA DE GRADUAÇÃO CURSO DE ENGENHARIA DA COMPUTAÇÃO

## PEDRO DELAVALD FRÁ

## PREVISÃO DE VENDAS:

Análise comparativa entre abordagens de aprendizado de máquina e Power BI

São Leopoldo

## PEDRO DELAVALD FRÁ

## PREVISÃO DE VENDAS:

Análise comparativa entre abordagens de aprendizado de máquina e Power BI

Trabalho de Conclusão de Curso apresentado  como  requisito  parcial  para obtenção do título de Bacharel em Engenharia da Computação, pelo Curso de Engenharia da Computação da Universidade  do  Vale  do  Rio  dos  Sinos (UNISINOS)

Orientador: Prof. MSc. Jean Schmith

## RESUMO

Este  trabalho  tem  como  objetivo  avaliar  e  comparar  o  desempenho  de diferentes  métodos  de  previsão  de  vendas,  utilizando  tanto  técnicas  estatísticas tradicionais  quanto  algoritmos  modernos  de  aprendizado  de  máquina,  aplicados  a dados reais de faturamento extraídos de um dashboard corporativo em Power BI. Diante  do  aumento  da  competitividade  e  da  demanda  por  decisões  empresariais baseadas em dados, destaca-se a necessidade de modelos preditivos cada vez mais precisos e robustos. O estudo envolve a implementação dos modelos ARIMA, Theta, Suavização Exponencial e XGBoost, analisando suas performances preditivas e as possibilidades de adoção dessas abordagens no contexto empresarial. Os resultados são avaliados a partir de métricas estatísticas padronizadas, permitindo identificar se algum modelo apresenta desempenho superior ao método atualmente empregado. A pesquisa contribui para a aproximação entre teoria e prática, oferecendo subsídios para  a  escolha  de  métodos  de  previsão  mais  adequados  às  necessidades  das organizações e potencializando o valor estratégico das análises de vendas.

Palavras-chave: Previsão de Vendas; Séries Temporais; Aprendizado de Máquina; Power  BI;  ARIMA;  XGBoost;  Suavização  Exponencial;  Método  Theta;  Business Intelligence.

## ABSTRACT

This  work  aims  to  evaluate  and  compare  the  performance  of  different  sales forecasting  methods,  employing  both  traditional  statistical  techniques  and  modern machine learning algorithms, applied to real revenue data extracted from a corporate dashboard in Power BI. Given the increasing competitiveness and demand for datadriven  business  decisions,  there  is  a  growing  need  for  more  accurate  and  robust predictive models. The study involves the implementation of ARIMA,  Theta, Exponential Smoothing, and XGBoost models, analyzing their predictive performance and the feasibility of adopting these approaches in corporate environments. The results are assessed using standardized statistical metrics, allowing for the identification of models that outperform the currently employed method. This research contributes to bridging the gap between theory and practice, offering guidance for the selection of forecasting  methods  that  best  fit  organizational  needs  and  enhancing  the  strategic value of sales analytics.

Key-words: Sales Forecasting; Time Series; Machine Learning; Power BI; ARIMA; XGBoost; Exponential Smoothing; Theta Method; Business Intelligence.

## LISTA DE FIGURAS

| Figura 1 - Metodologia geral do trabalho...................................................................30   |
|-----------------------------------------------------------------------------------------------------------------|
| Figura 2 - Metodologia do modelo ARIMA ................................................................34       |
| Figura 3 - Metodologia do modelo XGBoost.............................................................38         |
| Figura 4 - Metodologia do modelo de Suavização Exponencial................................42                    |
| Figura 5 - Metodologia do modelo Theta...................................................................45     |

## LISTA DE QUADROS

## LISTA DE SIGLAS

| CNN     | Convolutional Neural Network                       |
|---------|----------------------------------------------------|
| RNN     | Recurrent Neural Network                           |
| ARIMA   | Auto Regressive Integrated Moving Average          |
| XGBoost | X Gradient Boost                                   |
| ML      | Machine Learning                                   |
| PIB     | Produto Interno Bruto                              |
| SARIMA  | Seasonal Auto Regressive Integrated Moving Average |
| LSTM    | Long Short-Term Memory                             |
| STL     | Seasonal and Trend decomposition using LOESS       |
| AIC     | Akaike Information Criterion                       |
| AR      | Auto Regressive                                    |
| MA      | Moving Average                                     |
| SES     | Simple Exponential Smoothing                       |
| ACF     | Autocorrelation Function                           |
| PACF    | Parcial Autocorrelation Function                   |
| KPSS    | Kwiatkowski-Phillips-Schmidt-Shin                  |
| ADF     | Augmented Dickey Fuller                            |
| RMSSE   | Root Mean Squared Scaled Error                     |
| RMSE    | Root Mean Squared Error                            |
| MAE     | Mean Absolute Error                                |
| BI      | Business Intelligence                              |
| GBDT    | Gradient Boosting Decision Tree                    |

## SUMÁRIO

| 1 INTRODUÇÃO.......................................................................................................11        |                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| 1.1 TEMA..................................................................................................................11 |                                                                                                            |
| 1.2 DELIMITAÇÃO DO TEMA...................................................................................12                 |                                                                                                            |
| 1.3 PROBLEMA                                                                                                                 | ........................................................................................................12 |
| 1.4 OBJETIVOS........................................................................................................12      |                                                                                                            |
| 1.4.1 Objetivo geral                                                                                                         | .................................................................................................12        |
| 1.4.2 Objetivos específicos.....................................................................................12           |                                                                                                            |
| 1.5 JUSTIFICATIVA                                                                                                            | ..................................................................................................13       |
| 2 FUNDAMENTAÇÃO TEÓRICA.............................................................................14                       |                                                                                                            |
| 2.1 SÉRIES TEMPORAIS.........................................................................................14              |                                                                                                            |
| 2.1.1 Conceitos fundamentais e definições                                                                                    | ..........................................................14                                               |
| 2.1.2 Características principais..............................................................................14             |                                                                                                            |
| 2.1.3 Classificações de séries temporais..............................................................15                     |                                                                                                            |
| 2.1.4 Exemplos de aplicação..................................................................................16              |                                                                                                            |
| 2.2 MÉTODO THETA................................................................................................16           |                                                                                                            |
| 2.2.1 Descrição geral e origem...............................................................................17              |                                                                                                            |
| 2.2.2 Fundamentação teórica e parâmetros..........................................................17                         |                                                                                                            |
| 2.2.3 Equação da linha Theta .................................................................................18             |                                                                                                            |
| 2.2.4 Expressões aditivas e multiplicativas                                                                                  | ..........................................................18                                               |
| 2.2.5 Funcionamento do método para previsão de dados futuros                                                                 | .....................18                                                                                    |
| 2.2.6 Exemplos práticos de uso.............................................................................19                |                                                                                                            |
| 2.3 MODELO ARIMA                                                                                                             | ................................................................................................20         |
| 2.3.1 Definição e estrutura do modelo ARIMA......................................................20                          |                                                                                                            |
| 2.3.2 Conceitos e características do modelo ARIMA                                                                            | ...........................................21                                                              |
| 2.3.3 Como o modelo ARIMA funciona para prever dados futuros?                                                                | ..................21                                                                                       |
| 2.3.4 Casos práticos e exemplos na literatura......................................................22                        |                                                                                                            |
| 2.4 SUAVIZAÇÃO EXPONENCIAL...........................................................................23                      |                                                                                                            |
| 2.4.1 Definição e estrutura do método ..................................................................23                   |                                                                                                            |
| 2.4.2 Vantagens e limitações na previsão de dados                                                                            | ............................................24                                                             |
| 2.4.3 Aplicações e estudos de caso ......................................................................25                  |                                                                                                            |
| 2.5 XGBOOST...........................................................................................................26     |                                                                                                            |
| 2.5.1 Visão geral do Extreme Gradient Boosting..................................................26                           |                                                                                                            |

| 2.5.2 Características e conceitos do XGBoost .....................................................27                        |                                                                                                 |
|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| 2.5.3 Como o XGBoost prevê dados futuros ........................................................27                         |                                                                                                 |
| 2.5.4 Exemplos práticos de uso do XGBoost........................................................29                         |                                                                                                 |
| 3 METODOLOGIA....................................................................................................30         |                                                                                                 |
| 3.1 METODOLOGIA DE TRABALHO                                                                                                 | .......................................................................30                       |
| 3.1.1 Definição do problema e objetivos da previsão                                                                         | ..........................................31                                                    |
| 3.1.2 Coleta e integração dos dados                                                                                         | .....................................................................31                         |
| 3.1.3 Pré-processamento e transformações dos dados                                                                          | ......................................32                                                        |
| 3.1.4 Análise exploratória e estruturação da série temporal...............................32                                |                                                                                                 |
| 3.2 MODELOS DE PREVISÃO UTILIZADOS...........................................................33                             |                                                                                                 |
| 3.2.1 ARIMA..............................................................................................................34 |                                                                                                 |
| 3.2.1.1 Importação das bibliotecas e configuração do ambiente...............................35                              |                                                                                                 |
| 3.2.1.2 Ingestão e conversão dos dados para série temporal...................................35                             |                                                                                                 |
| 3.2.1.3 Verificação de estacionaridade e diferenciação                                                                      | ............................................35                                                  |
| 3.2.1.4 Divisão dos dados em conjuntos de treino e teste                                                                    | ........................................36                                                      |
| 3.2.1.5 Definição dos parâmetros p, d e q.................................................................36                |                                                                                                 |
| 3.2.1.6 Treinamento do modelo.................................................................................36            |                                                                                                 |
| 3.2.1.7 Validação do modelo e ajustes finos .............................................................37                 |                                                                                                 |
| 3.2.1.8 Análise residual                                                                                                    | .............................................................................................37 |
| 3.2.1.9 Armazenamento dos resultados para comparação futura.............................37                                  |                                                                                                 |
| 3.2.2 XGBoost..........................................................................................................38   |                                                                                                 |
| 3.2.2.1 Preparação e engenharia de variáveis..........................................................38                    |                                                                                                 |
| 3.2.2.2 Divisão dos dados em treino e teste                                                                                 | .............................................................39                                 |
| 3.2.2.3 Normalização e tratamento dos dados..........................................................39                     |                                                                                                 |
| 3.2.2.4 Configuração dos hiper parâmetros iniciais...................................................39                     |                                                                                                 |
| 3.2.2.5 Treinamento inicial do modelo.......................................................................40              |                                                                                                 |
| 3.2.2.6 Avaliação inicial de desempenho                                                                                     | ..................................................................41                            |
| 3.2.2.7 Busca e ajuste de hiper parâmetros..............................................................41                  |                                                                                                 |
| 3.2.2.8 Validação cruzada e análise de resultados                                                                           | ...................................................41                                           |
| 3.2.2.9 Geração das previsões finais e armazenamento dos resultados                                                         | ..................41                                                                            |
| 3.2.3 Suavização exponencial                                                                                                | ................................................................................42              |
| 3.2.3.1 Preparação dos dados                                                                                                | ..................................................................................42            |
| 3.2.3.2 Análise exploratória e estrutura da série temporal                                                                  | ........................................43                                                      |
| 3.2.3.3 Divisão em conjunto de treino e teste............................................................43                 |                                                                                                 |

10

| 3.2.3.4 Seleção do tipo de suavização exponencial e parâmetros............................43                                  |                                                                 |
|-------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| 3.2.3.5 Treinamento inicial do modelo.......................................................................44                |                                                                 |
| 3.2.3.6 Geração das previsões..................................................................................44             |                                                                 |
| 3.2.3.7 Avaliação do desempenho............................................................................44                 |                                                                 |
| 3.2.3.8 Ajuste fino e revalidação ...............................................................................44           |                                                                 |
| 3.2.3.9 Geração das previsões finais e armazenamento dos resultados                                                           | ..................44                                            |
| 3.2.4 Theta................................................................................................................45 |                                                                 |
| 3.2.4.1 Organização e pré-condições dos dados......................................................45                         |                                                                 |
| 3.2.4.2 Análise inicial e sazonalidade........................................................................46              |                                                                 |
| 3.2.4.3 Separação temporal para avaliação..............................................................46                     |                                                                 |
| 3.2.4.4 Configuração e execução do algoritmo .........................................................46                      |                                                                 |
| 3.2.4.5 Produção das previsões e pós-processamento.............................................46                             |                                                                 |
| 3.2.4.6 Avaliação quantitativa e diagnóstico                                                                                  | .............................................................47 |
| 3.2.4.7 Iteração e consolidação dos resultados                                                                                | ........................................................47      |
| 3.3 AVALIAÇÃO E COMPARAÇÃO DOS MODELOS..............................................47                                        |                                                                 |
| 3.4 CRONOGRAMA..................................................................................................48            |                                                                 |
| REFERÊNCIAS.........................................................................................................49        |                                                                 |

## 1 INTRODUÇÃO

A previsão de vendas, no contexto atual da transformação digital e da crescente demanda por decisões empresariais baseadas em dados, se estabelece como um dos grandes desafios e diferenciais competitivos para organizações de todos os portes. Com  mercados  cada  vez  mais  dinâmicos  e  suscetíveis  a  variações  econômicas, tecnológicas e comportamentais, a precisão nas estimativas de faturamento assume papel central no planejamento, controle de estoques, logística, definição de metas e estratégias comerciais. Este cenário impulsionou o avanço de diferentes métodos de previsão,  desde  técnicas  estatísticas  tradicionais  até  abordagens  inovadoras  de aprendizado  de  máquina,  que  vêm  transformando  a  forma  como  as  empresas analisam e projetam seus resultados futuros.

O uso disseminado de ferramentas de BI, como o Power BI, trouxe grandes avanços  para  a  visualização  e  interpretação  dos  dados  históricos  das  empresas, permitindo  a  elaboração  de  dashboards  customizados  para  acompanhamento  do desempenho de vendas. Contudo, muitos desses sistemas ainda utilizam métodos de previsão relativamente simples, que podem não captar integralmente a complexidade dos  padrões  temporais,  sazonalidades  e  variáveis  exógenas  presentes  nos  dados (ENSAFI et al., 2022). Paralelamente, algoritmos de ML, como o XGBoost, vêm sendo destacados na literatura por sua elevada acurácia preditiva, robustez e flexibilidade na incorporação de múltiplos fatores ao processo de modelagem,  sendo escolhido frequentemente em cenários reais e competições internacionais (CHEN; GUESTRIN, 2016).

Diante  desse  contexto,  torna-se  pertinente  avaliar,  sob  uma  perspectiva aplicada  e  comparativa,  se  modelos  de  ML  podem  efetivamente  aprimorar  as previsões  de  faturamento  realizadas  por  soluções  já  consolidadas  no  ambiente empresarial, como o Power BI, contribuindo para a geração de insights mais robustos e embasados para a tomada de decisão.

## 1.1 TEMA

O presente trabalho aborda o tema da previsão de vendas utilizando séries temporais,  com  foco  na  comparação  entre  métodos  tradicionais  e  modernos  de modelagem preditiva aplicados a dados reais de faturamento empresarial.

## 1.2 DELIMITAÇÃO DO TEMA

A pesquisa concentra-se na análise comparativa do desempenho de diferentes modelos de previsão utilizando dados históricos extraídos de um banco de dados. O estudo  limita-se  à  previsão  de  faturamento  mensal,  simulando  o  contexto  prático enfrentado por empresas que necessitam estimar o resultado do mês corrente com base em informações parciais, do primeiro dia do mês até o momento da consulta.

## 1.3 PROBLEMA

O problema que orienta este trabalho é: Modelos avançados de aprendizado de  máquina  podem  proporcionar  previsões  mais  precisas  de  faturamento,  quando comparados  à  abordagem  utilizada  em  dashboards  de  Power  BI?  A  investigação busca responder se a adoção de modelos de aprendizado de máquina como XGBoost, ARIMA,  Suavização  Exponencial  e  Theta  pode,  de  fato,  melhorar  a  acurácia  das projeções realizadas atualmente pela empresa, promovendo maior confiabilidade e valor estratégico às informações disponibilizadas.

## 1.4 OBJETIVOS

## 1.4.1 Objetivo geral

Avaliar,  de  forma  comparativa,  o  desempenho  de diferentes  abordagens  de previsão de vendas, sejam elas tradicionais ou baseadas em ML, aplicadas a dados reais  de  faturamento,  verificando  se  algum  dos  modelos  apresenta  desempenho superior ao método atualmente utilizado em dashboards de Power BI.

## 1.4.2 Objetivos específicos

- a)  Revisar  e  contextualizar  os  principais  conceitos  de  séries  temporais, métodos  estatísticos  clássicos  e  técnicas  de  ML  voltadas  à  previsão  de vendas, conforme descrito por autores como Bezerra (2006), Makridakis, Wheelwright e Hyndman (1999) e Ensafi et al. (2022);

- b)  Estruturar  e  pré-processar  os  dados históricos  de  faturamento  de  acordo com as exigências de cada modelo preditivo, assegurando anonimização, integridade e conformidade com boas práticas de ciência de dados;
- c)  Implementar, treinar e validar modelos de previsão ARIMA,  Theta, Suavização Exponencial e XGBoost, utilizando métricas estatísticas padronizadas para avaliação do desempenho;
- d)  Analisar comparativamente os resultados obtidos e discutir as vantagens, limitações e possibilidades práticas para adoção dos métodos preditivos no contexto empresarial.

Acredita-se que essa abordagem possibilite uma análise abrangente e rigorosa, identificando  as  oportunidades  e  desafios  envolvidos  na  transição  para  modelos preditivos mais avançados no ambiente corporativo.

## 1.5 JUSTIFICATIVA

A relevância deste estudo se justifica tanto pelo avanço recente das técnicas de análise preditiva quanto pela necessidade real de organizações aprimorarem seus processos de tomada de decisão frente a cenários de incerteza e competitividade. Do ponto  de  vista  acadêmico,  há  uma  lacuna  na  literatura  nacional  sobre  aplicações práticas  e  comparativas  de  modelos  de  machine  learning  em  ambientes  de  BI amplamente adotados por empresas brasileiras, como o Power BI (ENSAFI et al., 2022;  SHIRI  et  al.,  2024).  Internacionalmente,  pesquisas  vêm  demonstrando  o potencial  de  algoritmos  como  XGBoost  na  superação  de  métodos  tradicionais  de previsão, especialmente em séries temporais com padrões complexos e influências externas (CHEN; GUESTRIN, 2016).

No âmbito empresarial, a adoção de modelos mais precisos pode representar ganhos substanciais em  planejamento, controle financeiro e competitividade, permitindo que decisões sejam tomadas com maior base quantitativa e menor risco. Este  trabalho,  ao  propor  uma  análise  comparativa  fundamentada,  contribui  para aproximar a teoria e a prática, orientando gestores e profissionais de dados quanto à melhor escolha de métodos para suas demandas específicas.

## 2 FUNDAMENTAÇÃO TEÓRICA

Neste capítulo, apresenta-se o embasamento  teórico indispensável ao desenvolvimento  do  presente  estudo.  Serão  discutidos  os  conceitos  fundamentais relacionados à previsão de dados, contemplando tanto a aplicação de algoritmos de aprendizado de máquina quanto a utilização de cálculos no Power BI. A partir dessa fundamentação,  busca-se  sustentar  o  estudo  de  caso  realizado,  evidenciando  as principais  vantagens  e  limitações  de  cada  abordagem  na  análise  e  projeção  de informações.

## 2.1 SÉRIES TEMPORAIS

A análise de séries temporais é uma importante área da estatística, dedicada à compreensão, modelagem e previsão de fenômenos que são observados de forma sequencial  no  tempo.  Conforme  Bezerra  (2006),  a  utilização  da  análise  de  séries temporais é amplamente difundida em diversas áreas, como economia, meteorologia, saúde, controle de processos industriais, vendas e finanças, devido à capacidade de identificar padrões de comportamento e realizar previsões futuras com base em dados históricos.

## 2.1.1 Conceitos fundamentais e definições

De acordo com Parzen (1961), uma série temporal pode ser entendida como um  conjunto  de  observações  dispostas  cronologicamente,  sendo  representada matematicamente  como  um  processo  estocástico,  no  qual  cada  valor  observado corresponde a um instante específico no tempo.

## 2.1.2 Características principais

Entre os principais conceitos e características envolvidos na análise de séries temporais, destacam-se:

- a)  Estacionariedade:  Segundo  Bezerra  (2006),  a  estacionariedade  ocorre quando as propriedades estatísticas, tais como  média, variância e covariância,  permanecem  constantes  ao  longo  do  tempo.  A  condição  de

estacionariedade é importante para aplicação correta de diversos modelos, como os modelos ARIMA.

- b)  Tendência: Refere-se à direção predominante da série ao longo do tempo, podendo  ser  crescente,  decrescente  ou  estável.  Segundo  Makridakis, Wheelwright e Hyndman (1999), a tendência é fundamental para entender o comportamento das séries e escolher modelos adequados.
- c)  Sazonalidade: Corresponde  às  variações  periódicas  e  regulares  que ocorrem  em  intervalos  fixos,  como  mensal  ou  anual,  devido  a  fatores externos ou eventos recorrentes (MAKRIDAKIS, WHEELWRIGHT; HYNDMAN, 1999).
- d)  Autocorrelação:  Representa  a  correlação  da  série  consigo  mesma  em diferentes momentos do tempo (lags). De acordo com Parzen (1961), esse conceito é fundamental para identificar e compreender o comportamento das séries temporais.
- e) Ruído  branco:  Para  Bezerra  (2006),  é  a  parcela  aleatória  da  série temporal, composta por erros aleatórios independentes com média zero e variância constante, que não apresentam qualquer tipo de padrão previsível.

## 2.1.3 Classificações de séries temporais

Makridakis, Wheelwright e Hyndman (1999) classificam as séries temporais em tipos distintos:

- a)  Séries  estacionárias:  Caracterizam-se  por  apresentar  média  e  variância constantes ao longo do tempo. São frequentemente observadas em séries financeiras de retorno.
- b)  Séries não estacionárias: São séries cujas propriedades estatísticas, como média e/ou variância, alteram-se com o tempo. Exemplos comuns incluem séries econômicas como PIB e inflação.
- c)  Séries lineares e não lineares: Séries lineares podem ser modeladas por técnicas tradicionais,  como  ARIMA,  enquanto  séries não  lineares  exigem modelos mais avançados, como redes neurais artificiais (SHIRI et al., 2024).

## 2.1.4 Exemplos de aplicação

Vários  estudos  demonstram  a  aplicação  prática  das  séries  temporais  em diversos contextos:

- a)  Previsão  de  vendas  no  varejo:  Ensafi  et  al.  (2022)  compararam  técnicas tradicionais  como  SARIMA  e  Suavização  Exponencial  com  métodos avançados  como  redes  neurais  LSTM  e  CNN  para  previsão  das  vendas sazonais de móveis. Os resultados mostraram que as redes neurais LSTM apresentaram maior precisão na captura de padrões complexos e sazonais.
- b)  Previsão  de  vendas  semanais em lojas de departamento: Pao e Sullivan (2014) utilizaram técnicas como árvores de decisão, STL+ARIMA e redes neurais feed-forward com entradas temporais defasadas, concluindo que as redes neurais tiveram um desempenho superior, capturando com eficiência as sazonalidades das vendas semanais.
- c)  Aplicação  de  Deep  Learning  em  séries  temporais  complexas:  Shiri  et  al. (2024) realizaram uma revisão abrangente sobre o uso de modelos de deep learning, como CNN, RNN, LSTM e Transformer, em séries temporais. O estudo apontou que técnicas modernas baseadas em deep learning têm se mostrado  superiores  às  técnicas  tradicionais,  principalmente  em  séries complexas e com grandes volumes de dados.

## 2.2 MÉTODO THETA

O método Theta ganhou popularidade ao vencer a competição M3 de previsões de séries temporais devido à sua simplicidade e eficiência em gerar previsões precisas para  diversos  tipos  de  dados.  Desde  então,  este  método  tem  sido  amplamente estudado  e  aprimorado,  resultando  em  diferentes  variantes  que  exploram  seu potencial  para  aplicações  automáticas  e  mais  robustas.  (ASSIMAKOPOULOS; NIKILOPOULOS, 2000).

## 2.2.1 Descrição geral e origem

O método Theta é uma técnica de previsão uni variada que decompõe a série temporal original em componentes denominados "linhas Theta". Cada linha Theta é obtida  ajustandose  a  curvatura  dos  dados  originais  através  de  um  parâmetro  θ aplicado às segundas diferenças da série original. (ASSIMAKOPOULOS; NIKILOPOULOS,  2000;  SPILIOTIS;  ASSIMAKOPOULOS;  MAKRIDAKIS,  2020).  A combinação dessas linhas Theta gera previsões que equilibram tendências de curto e longo prazo. (ASSIMAKOPOULOS; NIKILOPOULOS, 2000).

## 2.2.2 Fundamentação teórica e parâmetros

As principais características do método Theta incluem:

- a)  Decomposição da série temporal: a série original é  dividida em múltiplas linhas  Theta,  destacando  diferentes  características  como  tendências  de curto e longo prazo (ASSIMAKOPOULOS; NIKOLOPOULOS, 2000).
- b) Parâmetro θ (Theta): controla a curvatura das linhas, com 𝜃 &lt; 1 enfatizando tendências de longo prazo e 𝜃 &gt; 1 destacando variações de curto prazo. (ASSIMAKOPOULOS; NIKOLOPOULOS, 2000; SPILIOTIS; ASSIMAKOPOULOS; MAKRIDAKIS, 2020).
- c)  Combinação de previsões: as previsões geradas a partir das linhas Theta são  combinadas  usando  ponderações  específicas  para  gerar  resultados mais robustos e precisos (FIORUCCI et al., 2016).
- d)  Flexibilidade  e  robustez:  permite  ajuste  e  adaptação  automática  dos parâmetros  para  diferentes  séries  temporais,  tornando-o  versátil  para diversos contextos (SPILIOTIS; ASSIMAKOPOULOS; MAKRIDAKIS, 2020).
- e)  Eficiência  computacional:  destaca-se  pela  simplicidade  computacional, sendo fácil e rápido de implementar, especialmente quando comparado com métodos mais complexos como ARIMA ou redes neurais (FIORUCCI et al., 2016).
- f)  Capacidade  de  generalização:  é  aplicável  em  séries  temporais  com diferentes  padrões,  como  tendências  lineares,  não  lineares,  séries  com

comportamento sazonal e séries irregulares (SPILIOTIS; ASSIMAKOPOULOS; MAKRIDAKIS, 2020).

- g)  Simplicidade na interpretação: oferece resultados facilmente interpretáveis, facilitando seu uso prático em  ambientes corporativos e industriais (FIORUCCI et al., 2016).

## 2.2.3 Equação da linha Theta

Segundo Spiliotis, Assimakopoulos e Makridakis (2020), o método Theta pode ser matematicamente descrito da seguinte forma:

Seja 𝑌 𝑡 uma série temporal observada no tempo 𝑡 .  Uma linha Theta 𝑍𝑡 (𝜃) é obtida pela expressão:

∇ 2 𝑍𝑡 (𝜃) = 𝜃∇ 2 𝑌 𝑡 = 𝜃(𝑌 𝑡 -2𝑌(𝑡-1) + 𝑌 (𝑡+2)), 𝑡 = 3, … , 𝑛 onde ∇ 2 𝑌 𝑡 é o operador das segundas diferenças da série original 𝑌 no ponto 𝑡 .

## 2.2.4 Expressões aditivas e multiplicativas

No método Theta, as previsões podem ser realizadas utilizando expressões aditivas ou multiplicativas:

- a)  Modelo aditivo: é o modelo original do método Theta, no qual as previsões são  obtidas  pela  combinação  linear  aditiva  das  linhas  Theta  ajustadas (ASSIMAKOPOULOS; NIKOLOPOULOS, 2000).
- b)  Modelo  multiplicativo:  é  uma  extensão  recente  do  método,  permitindo modelar situações em que componentes como sazonalidade e tendência interagem de forma multiplicativa, sendo especialmente útil em séries com tendência exponencial ou comportamento sazonal multiplicativo (SPILIOTIS; ASSIMAKOPOULOS; MAKRIDAKIS, 2020).

## 2.2.5 Funcionamento do método para previsão de dados futuros

Para  prever  dados  futuros,  o  método  Theta  realiza  as  seguintes  etapas (ASSIMAKOPOULOS; NIKOLOPOULOS, 2000; FIORUCCI, 2016):

- a)  Decomposição:  a  série  temporal  é  decomposta  em  linhas  Theta  com diferentes curvaturas.
- b)  Extrapolação:  cada  linha  é  extrapolada  individualmente,  frequentemente usando métodos simples, como suavização exponencial simples (SES) para tendências  de  curto  prazo  e  regressão  linear  para  tendências  de  longo prazo.
- c)  Combinação das linhas: as previsões individuais são combinadas, geralmente com pesos iguais ou otimizados, produzindo uma previsão final robusta.

## 2.2.6 Exemplos práticos de uso

- O método Theta tem sido amplamente aplicado em diversas áreas, demonstrando sua robustez:
- a)  Competição M3: a versão clássica do método Theta alcançou resultados superiores às demais técnicas na competição M3, uma famosa competição internacional focada em  métodos  de  previsão de séries temporais, especialmente em séries mensais e microeconômicas, destacando-se por sua precisão e simplicidade (MAKRIDAKIS; HIBON, 2000).
- b)  Diagnóstico automotivo: Lozia (2022) utilizou o método Theta na avaliação diagnóstica  de  amortecedores  automotivos,  demonstrando  a  eficácia  do método  em  modelar  e  prever  o  comportamento  dinâmico  de  sistemas mecânicos complexos.
- c)  Previsão automática: Spiliotis, Assimakopoulos e Makridakis (2020) propuseram  generalizações  do  método  Theta  capazes  de  selecionar automaticamente a forma mais apropriada (aditiva ou multiplicativa) e ajustar a inclinação das tendências, superando outros algoritmos automáticos em competições recentes (como M4), especialmente em séries anuais.

## 2.3 MODELO ARIMA

O modelo ARIMA é uma técnica estatística amplamente utilizada para análise e previsão de séries temporais, desenvolvido por Box  e  Jenkins (1970). É especialmente indicado para séries cujos valores passados e erros históricos podem ser utilizados para prever valores futuros (NEWBOLD, 1983).

## 2.3.1 Definição e estrutura do modelo ARIMA

O  modelo  ARIMA  é  uma  combinação  dos  modelos  autorregressivos  (AR), integrados (I) e de médias móveis (MA), definidos pela seguinte notação geral ARIMA (p, d, q), onde (NEWBOLD, 1983):

- a)  p: ordem do termo autorregressivo (AR), representa a relação linear entre a observação atual e as anteriores.
- b)  d: número de diferenciações necessárias para tornar a série estacionária.
- c)  q: ordem dos termos de média móvel (MA), que refletem os erros anteriores do modelo.

Matematicamente, o modelo ARIMA (p, d, q) pode ser expresso da seguinte forma (NEWBOLD, 1983):

<!-- formula-not-decoded -->

Onde:

- 𝑌 𝑡 : valor atual da série temporal.
- 𝑌 𝑡-1 , 𝑌 𝑡-2 ,..., 𝑌 𝑡-𝑝 : valores anteriores da série temporal (termos AR).
- 𝜀𝑡 :  erro  aleatório  (resíduos)  com distribuição normal, média zero e variância constante (ruído branco).
- 𝜀𝑡-1 , 𝜀𝑡-2 , ..., 𝜀𝑡-𝑞 : erros anteriores da série (termos MA).
- 𝛿 : constante.
- 𝜙1 , 𝜙2, … , 𝜙𝑝 : coeficientes do termo autorregressivo.
- 𝜃1 , 𝜃2, … , 𝜃 𝑞 : coeficientes do termo de média móvel.

## 2.3.2 Conceitos e características do modelo ARIMA

As principais características do modelo ARIMA incluem (BOX; JENKINS, 1970; FATTAH et al., 2018):

- a)  Flexibilidade:  Pode  ajustar-se  a  diversas  séries  temporais,  incorporando tendência, ciclos e sazonalidade.
- b)  Necessidade de estacionariedade: Séries temporais precisam ser estacionárias  para  utilização  correta  do  modelo.  A  estacionariedade  é geralmente obtida por diferenciação sucessiva das séries temporais.
- c)  Simplicidade: Fácil de compreender e implementar, apresentando resultados robustos em previsões de curto prazo.

Para verificar se uma série é estacionária, frequentemente são utilizados testes estatísticos como o teste Dickey-Fuller (ADF) e o teste KPSS (MURAT et al., 2018).

## 2.3.3 Como o modelo ARIMA funciona para prever dados futuros?

O  processo  de  construção  do  modelo  ARIMA  segue  a  metodologia  BoxJenkins,  que  possui  as  seguintes  etapas  (BOX;  JENKINS,  1970;  MONDAL  et  al., 2014):

- a)  Identificação do modelo: Determinação das ordens p, d e q, com base na análise  gráfica  das  funções  de  autocorrelação  (ACF)  e  autocorrelação parcial (PACF).
- b)  Estimação  dos  parâmetros:  Os  coeficientes  do  modelo  são  estimados, normalmente utilizando o método da máxima verossimilhança.
- c)  Diagnóstico do modelo: Verificação da adequação do modelo por meio da análise dos resíduos (erros), usando testes como o teste de Ljung-Box e critérios estatísticos como AIC (Critério de Informação de Akaike).
- d)  Previsão:  Realização  da  previsão  de  valores  futuros  utilizando  o  modelo ajustado.

## 2.3.4 Casos práticos e exemplos na literatura

- O  modelo  ARIMA  tem  diversas  aplicações  práticas,  como  evidenciado  em diferentes estudos acadêmicos:
- a)  Previsão  de  demanda  em  indústrias  alimentícias:  Fattah  et  al.  (2018) mostraram que o modelo ARIMA (1,0,1) foi eficaz em prever a demanda futura, ajudando a empresa na gestão eficiente de estoques e redução de custos.
- b)  Previsão  de  vendas  no  e-commerce:  Um  modelo  híbrido  combinando ARIMA  com  redes  neurais  LSTM  foi  utilizado  para  previsão  precisa  em ambientes com alta volatilidade, como o comércio eletrônico (VAVLIAKIS et al., 2021).
- c)  Previsão no mercado farmacêutico: Fourkiotis e Tsadiras (2024) utilizaram ARIMA  em  combinação  com  técnicas  de  aprendizado  de  máquina  para prever a demanda por produtos farmacêuticos, mostrando sua eficácia em capturar efeitos sazonais. Para enfrentar esse desafio, Fourkiotis e Tsadiras (2024) utilizaram técnicas de análise uni variada de séries temporais para desenvolver previsões mais precisas. Os autores analisaram uma base de dados real contendo 600.000 registros históricos de vendas provenientes de uma  farmácia  online,  abrangendo  um  período  entre  2014  e  2019.  A metodologia proposta envolveu as etapas de pré-processamento e limpeza de dados, segmentação dos dados, análise exploratória e identificação dos padrões  temporais,  aplicação  e  comparação  do  modelo  ARIMA  com modelos avançados de ML como LSTM e XGBoost e, por fim, avaliação do modelo  com  métricas  específicas.  Os  resultados  demonstraram  que  o modelo  ARIMA  apresentou  uma  boa  capacidade  preditiva  ao  capturar adequadamente a sazonalidade e tendências lineares de vendas. Contudo, os  autores  destacaram  que modelos de ML avançados, especialmente o XGBoost, tiveram um desempenho ainda superior. Em particular, o XGBoost obteve as menores taxas de erro absoluto percentual médio (MAPE). Apesar da  boa  performance  dos  modelos  avançados  de  Machine  Learning,  o modelo  ARIMA  ainda  obteve  desempenho  competitivo  e  foi  considerado

eficaz especialmente em séries temporais com forte componente linear e sazonalidade bem definida.

- d)  Previsão de preços no mercado financeiro: Mondal et al. (2014) utilizaram ARIMA  para  prever  preços  de  ações,  destacando  sua  simplicidade  e robustez na previsão de tendências.

## 2.4 SUAVIZAÇÃO EXPONENCIAL

O método de suavização exponencial tem recebido grande atenção no contexto de  previsões  estatísticas  devido  à  sua  eficácia,  simplicidade  e  adaptabilidade  na previsão de séries temporais. Sua popularidade advém da capacidade intrínseca de atribuir pesos maiores às observações mais recentes em detrimento das observações mais antigas, permitindo rápidas adaptações às mudanças na dinâmica dos dados (GARDNER, 1985; CIPRA, 1992).

Essa técnica tornou-se uma abordagem padrão em diversos campos práticos, incluindo gestão de estoques, controle de processos industriais, finanças e gestão de cadeias de suprimentos. Sua ampla adoção se dá pela facilidade computacional e pela interpretação de suas previsões em comparação com métodos mais complexos como modelos ARIMA e redes neurais (MCKENZIE, 1984).

## 2.4.1 Definição e estrutura do método

O método de exponential smoothing é uma técnica recursiva para previsão de séries  temporais  que  se  baseia  na  ponderação  exponencial  decrescente  das observações passadas. Formalmente, uma previsão futura é construída como uma combinação linear entre a observação mais recente e a previsão feita anteriormente. Essa  característica de  atualização recursiva  confere simplicidade  e eficiência computacional ao método (BROWN, 1962; MCKENZIE, 1984).

Matematicamente, para o SES, a previsão do valor da série temporal 𝑋𝑡+1 pode ser expressa por:

<!-- formula-not-decoded -->

Onde:

- 𝑋 ̂ 𝑡+1 : valor previsto para o próximo período;

- 𝑋𝑡 : valor observado no período atual;
- 𝑋 ̂ 𝑡 : previsão feita anteriormente para o período atual;
- 𝛼 :  constante  de  suavização 0 &lt; 𝛼 &lt; 1 ,  que  define  o  grau  de  ponderação aplicado ao dado mais recente (BROWN, 1962).

Já  métodos  mais  avançados,  como  o  método  de  Holt-Winters,  consideram explicitamente os componentes de nível, tendência e sazonalidade da série temporal. Segundo Gardner (1985), para séries com comportamento sazonal e tendência linear, a previsão futura para ℎ passos à frente é dada pela expressão geral do método de Holt-Winters multiplicativo:

<!-- formula-not-decoded -->

Onde:

- 𝐿𝑡 é o nível estimado da série no tempo 𝑡 ;
- 𝑏𝑡 é a tendência estimada no tempo 𝑡 ;
- 𝑆𝑡+ℎ-𝑚(𝑘+1) é o fator sazonal estimado no tempo correspondente;
- ℎ representa o horizonte futuro da previsão (quantidade de períodos à frente);
- 𝑚 é  o  período  sazonal  da  série  (por  exemplo, 𝑚 = 12 para  séries  mensais anuais);
- 𝑘 é o número de ciclos completos transcorridos.

Esses  métodos  avançados  permitem  previsões  mais  precisas  em  séries complexas, com tendências claras ou padrões sazonais fortes, superando métodos mais  simples  como  médias  móveis  ou  o  próprio  exponential  smoothing  simples (MCKENZIE, 1984; GARDNER, 1985).

## 2.4.2 Vantagens e limitações na previsão de dados

Entre  as  características  fundamentais  do  método  de  exponential  smoothing destacam-se:

- a)  Adaptabilidade: capacidade de responder rapidamente às alterações estruturais  na  série  temporal,  atribuindo  pesos  exponenciais  aos  dados recentes (GARDNER, 1985).

- b)  Simplicidade  computacional:  a  estrutura  recursiva  dos  cálculos  torna  o método atrativo em aplicações práticas, especialmente onde é necessária atualização constante das previsões (BROWN, 1962).
- c)  Flexibilidade  estrutural:  diferentes  versões,  como  simples,  dupla  e  tripla (Holt-Winters),  permitem  modelar  comportamentos  como  tendência  e sazonalidade com eficiência (MCKENZIE, 1984).
- d)  Robustez:  versões  robustas  do  método,  que  usam  a  minimização  dos desvios  absolutos  ou  métodos  M-estimadores  ao  invés  de  mínimos quadrados, têm maior resistência a dados atípicos e séries temporais com distribuições assimétricas ou de caudas pesadas (CIPRA, 1992).

## 2.4.3 Aplicações e estudos de caso

- a)  Impacto  da  suavização  exponencial  no  Efeito  Bullwhip:  Chen,  Ryan  e Simchi-Levi (2000) investigaram como a utilização do exponential smoothing na previsão de demanda pode intensificar o efeito bullwhip, fenômeno no qual pequenas variações na demanda são ampliadas ao longo da cadeia de suprimentos. Eles demonstraram que, ao utilizar previsões com exponential smoothing,  as  variações  nas  demandas  observadas  pelos  fabricantes  se tornam significativamente maiores do que as percebidas pelos varejistas, aumentando os desafios de gestão e planejamento logístico nas organizações.
- b)  Robustez a outliers em séries temporais: Cipra (1992) avaliou o desempenho de versões robustas do método de exponential smoothing em séries temporais contaminadas por outliers e distribuições de caudas longas. Utilizando  minimização  dos  desvios  absolutos  (norma 𝐿1 )  em  vez  dos mínimos quadrados, Cipra verificou experimentalmente que essas versões robustas forneceram previsões significativamente mais estáveis e precisas na presença de valores extremos, superando métodos tradicionais especialmente em séries financeiras e industriais onde valores atípicos são comuns.
- c)  Aplicações em controle de estoques: Gardner (1985) destacou o uso bemsucedido de exponential smoothing no controle e previsão para gestão de estoques. Nesse contexto, foram aplicadas variações do método para prever

demandas futuras e determinar níveis ótimos de estoque, reduzindo custos relacionados  à  manutenção  excessiva  ou  insuficiente  de  produtos  em inventário.  Esse  exemplo  demonstra  claramente  como  o  exponential smoothing  pode  auxiliar  gestores  a  otimizarem  recursos  financeiros  e logísticos nas organizações.

- d)  Previsões  de  demanda  em  séries  sazonais  e  com  tendência:  McKenzie (1984) apresentou exemplos práticos demonstrando a eficácia do exponential  smoothing  para  séries  temporais  com  forte  comportamento sazonal e tendência definida. Em seu estudo, foi utilizado o método HoltWinters para capturar esses componentes, proporcionando previsões mais precisas que outros métodos tradicionais como médias móveis simples e modelos  ARIMA  em  séries  complexas,  especialmente  no  contexto  de demanda sazonal de varejo e setores produtivos.

## 2.5 XGBOOST

O XGBoost tornou-se um dos métodos mais populares e eficazes no âmbito da previsão  e  classificação  em  machine  learning,  devido  à  sua  capacidade  de  lidar eficientemente  com  grandes  quantidades  de  dados  e  produzir  modelos  altamente precisos. Originalmente proposto por Chen e Guestrin em 2016, o XGBoost combina otimizações  algorítmicas  e  técnicas  avançadas  de  engenharia  de  sistemas  para aprimorar significativamente o desempenho de previsões e classificações em diversas áreas (CHEN; GUESTRIN, 2016).

## 2.5.1 Visão geral do Extreme Gradient Boosting

O XGBoost é uma implementação otimizada do algoritmo Gradient Boosting, baseado  em  árvores  de  decisão  sequenciais.  Diferentemente  das  abordagens tradicionais, que utilizam árvores independentes (como o Random Forest), o XGBoost constrói árvores de maneira iterativa, com cada árvore subsequente aprendendo dos resíduos e erros das anteriores. A combinação final das árvores resulta em um modelo robusto  e  altamente  eficiente  para  prever  valores  futuros  e  classificar  dados complexos (MALIK; HARODE; KUNWAR, 2020).

## 2.5.2 Características e conceitos do XGBoost

Entre as características fundamentais do XGBoost destacam-se:

- a)  Boosting: Método de aprendizado de máquina que cria um modelo forte por meio da combinação sequencial de modelos fracos. Cada novo modelo tenta corrigir  os  erros  dos  modelos  anteriores  (MALIK;  HARODE;  KUNWAR, 2020).
- b)  Regularização: O XGBoost incorpora penalidades ao modelo para evitar o ajuste excessivo (overfitting), limitando a complexidade através de parâmetros  como  profundidade  máxima  das  árvores,  penalização  por complexidade  (gamma)  e  regularização  dos  pesos  das  folhas  (lambda). Essa abordagem resulta em modelos mais generalizáveis (CHEN; GUESTRIN, 2016).
- c)  Sparsity-aware  Split  Finding:  Um  algoritmo  que  otimiza  o  processo  de divisão das árvores levando  em conta a esparsidade dos dados, economizando  recursos  computacionais  ao  ignorar  valores  ausentes  ou zerados durante a construção das árvores (CHEN; GUESTRIN, 2016).
- d)  Paralelização  e  computação  distribuída:  O XGBoost é projetado para ser executado em múltiplas  CPUs,  permitindo o  processamento  paralelo  dos dados e acelerando significativamente o treinamento de grandes modelos (CHEN; GUESTRIN, 2016).
- e)  Shrinking  e  Column  Subsampling:  Técnicas  adicionais  que  ajudam  a controlar a complexidade do modelo. Shrinking reduz o impacto individual de cada árvore, enquanto Column Subsampling seleciona aleatoriamente um subconjunto de atributos para cada árvore, aumentando a robustez e a velocidade do modelo (CHEN; GUESTRIN, 2016).

## 2.5.3 Como o XGBoost prevê dados futuros

- O  funcionamento  do  XGBoost  para  previsões  ocorre  de  maneira  iterativa, seguindo os passos:

- a)  Inicialização: O processo se inicia com a definição de uma previsão inicial, que geralmente corresponde à média dos valores reais presentes nos dados de treinamento, no caso de problemas de regressão. Essa previsão inicial serve como ponto de partida para o modelo e representa a estimativa mais simples  possível  sem  considerar  ainda  as  relações  complexas  entre  as variáveis (CHEN; GUESTRIN, 2016; NIELSEN, 2016).
- b)  Cálculo  dos  resíduos:  Após  a  obtenção  da  previsão  inicial,  calcula-se  a diferença entre os valores previstos e os valores reais, gerando assim os resíduos. Esses resíduos indicam o quanto o modelo atual está errando na previsão.  O  objetivo  do  XGBoost  é  reduzir  esses  resíduos  a  cada  nova iteração, corrigindo gradualmente as falhas do modelo anterior (NIELSEN, 2016; ZHANG et al., 2021).
- c)  Treinamento iterativo das árvores: Em cada iteração, uma nova árvore de decisão é treinada, não para prever diretamente os valores finais, mas sim para modelar os resíduos obtidos na etapa anterior. Ou seja, cada árvore seguinte  busca  aprender  e  corrigir  os  erros  cometidos  pelo  conjunto  das árvores  anteriores,  ajustando-se  a  padrões  ainda  não  capturados  (XIE; ZHANG, 2021; NIELSEN, 2016).
- d)  Atualização  das  previsões:  As  previsões  do  modelo  são  atualizadas somando as previsões das novas árvores treinadas às previsões acumuladas das árvores anteriores. Com  isso, o modelo torna-se progressivamente mais preciso a cada ciclo, pois incorpora sucessivamente correções  dos  erros  passados.  Ao  final  do  processo,  a  previsão  final  é composta  pela  soma  ponderada  de  todas  as  árvores  criadas  durante  as iterações, representando assim uma combinação de múltiplos aprendizados parciais (CHEN; GUESTRIN, 2016; XIE; ZHANG, 2021).

A função objetivo otimizada no processo é:

<!-- formula-not-decoded -->

onde:

𝑙(𝑦 ̂ 𝑦 , 𝑦 𝑖 ) representa a função de perda (e.g., erro quadrático médio);

Ω(𝑓 𝑘 ) representa o termo de regularização que controla a complexidade do modelo (CHEN; GUESTRIN, 2016).

## 2.5.4 Exemplos práticos de uso do XGBoost

- a)  Utilidades: Segundo Noorunnahar et al. (apud Kontopoulou et al., 2023), no campo de utilidades, foi conduzido um estudo com o objetivo de prever a produção  anual  de  arroz  em  Bangladesh.  Os  autores  compararam  a precisão das previsões feitas por um método ARIMA otimizado, fundamentado no critério AIC, e pelo algoritmo XGBoost. Para a avaliação dos modelos, foram consideradas métricas de erro como MAE, MPE, RMSE e  MAPE.  Os  resultados  indicaram  que  o  modelo  XGBoost  obteve  um desempenho  superior  em  relação  ao  ARIMA  no  conjunto  de  teste, demonstrando  maior  eficácia  na  previsão  da  produção  de  arroz  para  o contexto analisado.
- b)  Previsão de volume de vendas no varejo: No setor de utilidades e comércio, o XGBoost tem se mostrado eficaz na previsão de volumes de vendas. A pesquisa de Dairu e Shilong (2021) é um exemplo, onde o modelo XGBoost foi utilizado para prever o volume de vendas no varejo, comparando seus resultados com o ARIMA clássico, o algoritmo GBDT, um modelo de LSTM e a ferramenta de previsão Prophet. Os resultados desse estudo indicaram que as abordagens baseadas em árvores, treinadas com características de clima e temperatura, ofereceram o melhor desempenho de previsão entre os cinco  modelos,  enquanto  o  ARIMA  apresentou  o  pior  desempenho. Notavelmente,  o  XGBoost  exigiu  significativamente  menos  iterações  de treinamento  do  que  o  GBDT  e,  juntamente  com  o  GBDT,  necessitou  de menos dados e recursos em contraste com os modelos de  LSTM. Além disso, os autores propuseram um modelo de previsão de vendas baseado em XGBoost para um conjunto de dados de bens de varejo do Walmart, demonstrando  bom  desempenho  com  menor  tempo  de  computação  e recursos de memória.

## 3 METODOLOGIA

Este  capítulo  apresenta  os  procedimentos  metodológicos  adotados  para  a realização  da  presente  pesquisa,  detalhando  de  forma  sistemática  as  etapas  que orientaram  o  desenvolvimento  do  estudo.  São  descritos  o  tipo  de  pesquisa,  a abordagem utilizada, os métodos de coleta e análise dos dados, bem como os critérios que fundamentaram as escolhas metodológicas. O objetivo é conferir transparência e fundamentação  científica  ao  percurso  investigativo,  garantindo  a  validade  e  a confiabilidade dos resultados obtidos.

## 3.1 METODOLOGIA DE TRABALHO

Com  o  intuito  de  proporcionar  uma  visão  geral  do  percurso  metodológico adotado, a figura a seguir apresenta, de forma esquemática, as principais etapas e procedimentos desenvolvidos ao longo deste trabalho. O diagrama tem como objetivo ilustrar,  de  maneira clara e objetiva, a estrutura metodológica geral que orientou a condução da pesquisa.

Figura 1 - Metodologia geral do trabalho

<!-- image -->

Fonte: elaborado pelo autor

## 3.1.1 Definição do problema e objetivos da previsão

Este trabalho tem como ponto de partida uma necessidade prática observada em um dos  produtos  desenvolvidos  pela  empresa  onde  atuo,  voltado  à  análise  e visualização  de  dados  corporativos.  Especificamente,  trata-se  de  um  dashboard construído na plataforma Power BI, que apresenta diversas análises de desempenho, incluindo uma medida responsável por estimar o faturamento do mês corrente com base nos dados registrados desde o primeiro dia do mês até o momento da consulta.

O problema que este trabalho propõe a investigar consiste em avaliar se é possível aprimorar essa estimativa por meio da aplicação de modelos de aprendizado de máquina. Para isso, serão desenvolvidos diferentes modelos preditivos utilizando os mesmos dados utilizados atualmente no dashboard, buscando simular o contexto real de previsão. Em seguida, será avaliado o desempenho de cada modelo com base em  métricas  estatísticas,  e  comparado  o  resultado  mais  eficaz  com  a  previsão atualmente gerada pelo Power BI.

O objetivo principal deste estudo é verificar se algum dos modelos testados apresenta desempenho superior ao cálculo de previsão utilizado hoje no produto da empresa. Caso isso ocorra, a adoção do modelo poderá resultar em previsões mais precisas e na geração de insights mais robustos e estratégicos.

## 3.1.2 Coleta e integração dos dados

A  coleta  e  integração  dos  dados  utilizados  neste  trabalho  serão  realizadas através da ferramenta Visual Studio Code. Os dados empregados correspondem às séries  históricas  de  faturamento  disponíveis  em  um  produto  interno  da  empresa, sendo originalmente utilizados em um dashboard desenvolvido em Power BI.

Inicialmente,  será  realizada  a  ingestão  dos  dados  em  seu  formato  bruto, assegurando a preservação de todas as informações relevantes. Considerando os aspectos éticos e a necessidade de garantir a confidencialidade das informações, será feita  a  anonimização  dos  dados  diretamente  no  Visual  Studio  Code,  por  meio  de transformações  utilizando  a  linguagem  de  programação  Python.  Essa  etapa  será essencial para remover ou ofuscar quaisquer identificadores sensíveis,  sem comprometer a estrutura ou a qualidade dos dados utilizados nas análises.

Após  a  anonimização,  os  dados  serão  estruturados  e  armazenados  na plataforma  para  dar  continuidade  às  etapas  de  pré-processamento,  modelagem  e validação, mantendo total alinhamento com a granularidade temporal e o contexto operacional do sistema de previsão atualmente existente.

## 3.1.3 Pré-processamento e transformações dos dados

Após a coleta dos dados, será iniciado o processo de pré-processamento com o objetivo de preparar as informações para a aplicação dos modelos de previsão. Esta etapa  compreenderá  tanto  procedimentos  voltados  à  integridade  e  à  estrutura analítica dos dados quanto à conformidade ética e à proteção da privacidade.

Como primeira medida, será realizada a anonimização dos dados diretamente na ferramenta Visual Studio Code, utilizando scripts desenvolvidos em Python. Essa anonimização terá como finalidade remover ou substituir identificadores sensíveis por pseudônimos ou códigos aleatórios, assegurando que nenhuma informação de caráter pessoal ou sigiloso pudesse ser associada a entidades reais. Essa prática está em conformidade com os princípios de segurança de dados e com a responsabilidade profissional no manuseio de informações empresariais.

Concluída a anonimização, serão conduzidas as etapas de transformação dos dados.  Os  registros  serão  organizados  em  formato  de  série  temporal  uni  variada, respeitando a ordenação cronológica e a granularidade mensal original dos dados de faturamento. Será realizado o tratamento de valores ausentes por meio de técnicas apropriadas, como interpolação linear ou imputação por média móvel, a depender da distribuição local dos dados.

Outras etapas incluirão a padronização dos tipos de dados, normalização de nomes de colunas, e a identificação e tratamento de valores atípicos (outliers), que poderiam interferir na qualidade das previsões. Ao final desse processo, será gerado um  conjunto  de  dados  estruturado  e  limpo,  que  servirá  como  base  geral  para  a construção das versões específicas adaptadas a cada modelo preditivo.

## 3.1.4 Análise exploratória e estruturação da série temporal

Com os dados devidamente pré-processados e estruturados, se iniciará a etapa de análise exploratória, com o objetivo de compreender o comportamento histórico da

série  de  faturamento  e  identificar  padrões  relevantes  para  o  desenvolvimento  dos modelos preditivos.

Primeiramente,  serão  construídas  visualizações  gráficas  da  série  temporal, como linhas de tendência, histogramas de distribuição e gráficos de decomposição, a fim  de  observar  aspectos  como  crescimento  ao  longo  do  tempo,  presença  de sazonalidade, variações abruptas e possíveis rupturas na estrutura dos dados. Essa análise  visual  é  essencial  para  a  identificação  inicial  de  tendências  e  ciclos econômicos característicos do contexto empresarial.

Em seguida, serão aplicadas técnicas estatísticas para quantificar e validar os padrões observados. Serão calculadas medidas descritivas como média, mediana, desvio-padrão e coeficiente de variação, além da aplicação de testes formais para verificação de estacionariedade, como os testes ADF e KPSS. Os resultados desses testes  permitirão  avaliar  a  necessidade  de diferenciação  ou outras  transformações específicas para tornar a série adequada aos requisitos dos modelos estatísticos.

Além  disso,  serão  realizadas  análises  de  autocorrelação  e  autocorrelação parcial (ACF e PACF), fundamentais para a parametrização de modelos como ARIMA, além da identificação de lags relevantes.

Com base nos resultados desta análise, a série temporal será estruturada em diferentes formatos para atender às exigências de cada abordagem preditiva.

## 3.2 MODELOS DE PREVISÃO UTILIZADOS

A modelagem preditiva é a etapa central deste trabalho, sendo responsável por transformar os dados estruturados em previsões quantitativas para o faturamento do produto  analisado.  Considerando  as  diferentes  abordagens  e  características  dos dados, serão selecionados múltiplos modelos de previsão, cada um com suas próprias vantagens, desvantagens e requisitos específicos de pré-processamento.

Os modelos escolhidos para este estudo incluem técnicas tradicionais de séries temporais, como ARIMA e Theta, bem como algoritmos mais recentes e avançados, como  XGBoost,  que  são  amplamente  utilizados  em  aplicações  empresariais  para problemas de previsão com séries temporais. Cada um desses modelos foi avaliado quanto à sua capacidade de capturar padrões históricos, prever tendências futuras e lidar  com os desafios típicos desse tipo de dado, como sazonalidade, tendência e variações irregulares.

Para  garantir  uma  análise  comparativa  robusta,  foram  considerados  fatores como a facilidade de implementação, complexidade computacional e a precisão das previsões geradas. Além disso, cada modelo será treinado e validado com os mesmos conjuntos de dados, permitindo uma comparação justa e direta de seu desempenho.

Nos subtópicos a seguir, cada modelo é apresentado individualmente, incluindo os requisitos específicos para pré-processamento dos dados e o diagrama do fluxo metodológico correspondente.

## 3.2.1 ARIMA

A figura a seguir mostra a metodologia utilizada para o modelo.

Figura 2 - Metodologia do modelo ARIMA

<!-- image -->

Fonte: elaborado pelo autor

## 3.2.1.1 Importação das bibliotecas e configuração do ambiente

Para iniciar a modelagem com ARIMA, serão usadas três bibliotecas Python, sendo elas a biblioteca Darts, que contém as várias funções já pré-programadas para processamento  de  séries  temporais,  Pandas  para  manipulação  de  dos  dados  e Matplotlib para a criação de gráficos que auxiliarão na análise exploratória e validação dos modelos.

Essa configuração do ambiente é uma etapa fundamental, pois garante que todas as operações subsequentes, como treinamento do modelo e cálculo de métricas possam ser realizadas de forma eficiente e organizada.

## 3.2.1.2 Ingestão e conversão dos dados para série temporal

Após a configuração do ambiente, o próximo passo será carregar os dados que serão  utilizados  no  modelo.  Para  isso,  será  necessário  que  os  dados  estejam estruturados de forma adequada, com uma coluna de datas como índice. No Darts, os dados precisam ser convertidos para a estrutura TimeSeries, que é otimizada para operações típicas de séries temporais, como diferenciação, transformação e predição.

Essa conversão é crucial, pois permite que o modelo ARIMA manipule os dados corretamente e aplique funções especializadas para séries temporais, como detecção de sazonalidade, análise de tendência e validação cruzada.

## 3.2.1.3 Verificação de estacionaridade e diferenciação

A  estacionaridade  é  uma  premissa  essencial  para  a  aplicação  do  modelo ARIMA,  pois  pressupõe  que  as  propriedades  estatísticas  da  série,  como  média  e variância,  permaneçam constantes ao longo do tempo. Para verificar se a série é estacionária, serão aplicados os testes ADF e o KPSS, que avaliam a presença de tendências e sazonalidades.

Se os testes indicarem que a série não é estacionária, será necessário aplicar uma transformação de diferenciação para estabilizar a média e remover a tendência. Esse  processo  é  importante  para  evitar  que  o  modelo  ajuste  padrões  espúrios, garantindo previsões mais precisas e robustas.

## 3.2.1.4 Divisão dos dados em conjuntos de treino e teste

Para  validar  adequadamente  o  modelo,  é  fundamental  dividir  os  dados  em conjuntos de treino e teste. Essa divisão deverá respeitar a ordem temporal dos dados, para  que  as  previsões  do  modelo  sejam  baseadas  em  informações  passadas, simulando um cenário real de produção. Será adotado inicialmente a prática comum de utilizar 80% dos dados para treino e 20% para teste, mas essa proporção poderá ser ajustada futuramente.

A separação dos dados desta forma permite avaliar a capacidade do modelo de generalizar para novos dados e evitar o risco de overfitting, onde o modelo se ajusta excessivamente ao conjunto de treino e perde desempenho em dados desconhecidos.

## 3.2.1.5 Definição dos parâmetros p, d e q

Os parâmetros do modelo ARIMA, p, d e q, serão definidos para que o modelo capture corretamente os padrões da série temporal. Esses parâmetros representam:

- a)  p  (ordem autorregressiva): Número de lags utilizados para prever o valor atual com base em valores passados.
- b)  d  (ordem  de  diferenciação):  Número  de  vezes  que  a  série  deve  ser diferenciada para remover tendências.
- c)  q (ordem do modelo de média móvel): Número de termos de erro defasados usados para ajustar o modelo.

A escolha desses parâmetros será feita com base na análise das funções de autocorrelação  (ACF)  e  autocorrelação  parcial  (PACF),  que  indicam  a  força  e  o alcance das correlações temporais presentes nos dados. Esta é uma etapa crítica, pois  define  a  complexidade  do  modelo  e  sua  capacidade  de  capturar  padrões temporais.

## 3.2.1.6 Treinamento do modelo

Com os parâmetros definidos, o próximo passo será ajustar o modelo ARIMA aos  dados  de  treino.  Esse  ajuste  será  feito  utilizando  o  método  de  máxima

verossimilhança, que busca encontrar os coeficientes do modelo que minimizam o erro de previsão no conjunto de treino.

Durante esta etapa, o modelo aprende os padrões históricos da série, ajustando os termos autorregressivos, de diferenciação e de média móvel para fornecer a melhor representação possível dos dados passados.

## 3.2.1.7 Validação do modelo e ajustes finos

Após o treinamento inicial, será necessário validar o modelo no conjunto de teste para avaliar sua capacidade de generalização. Esta etapa será feita a partir do cálculo de métricas como MAE e RMSE, que medem a precisão das previsões.

Além disso, será analisada a possibilidade de aplicar técnicas de ajuste fino, como grid search, para refinar os parâmetros p, d e q e otimizar o desempenho do modelo. Esse ajuste é essencial para garantir que o modelo não apenas ajuste bem os dados de treino, mas também seja capaz de fazer previsões precisas em dados novos.

## 3.2.1.8 Análise residual

Uma análise dos resíduos do modelo será feita para verificar se os erros de previsão se distribuem de forma aleatória, sem padrões não modelados. Resíduos com  padrões  indicam  que  o  modelo  não  conseguiu  capturar  completamente  as relações temporais nos dados, sugerindo a necessidade de ajustes nos parâmetros.

Além disso, essa análise também tem o intuito de revelar a presença de outliers ou  eventos  atípicos  que  não  foram  adequadamente  modelados,  o  que  pode comprometer a precisão das previsões futuras.

## 3.2.1.9 Armazenamento dos resultados para comparação futura

Finalmente,  os  resultados  do  modelo  ARIMA,  incluindo  as  previsões  e  os resíduos, serão salvos para futura comparação com os demais modelos e com as previsões atualmente geradas pelo Power BI. Essa etapa é essencial para a análise final do desempenho do modelo e para a escolha da abordagem preditiva mais precisa e robusta para o objetivo final.

## 3.2.2 XGBoost

A figura 3 mostra a metodologia utilizada para o modelo.

Figura 3 -Metodologia do modelo XGBoost

<!-- image -->

Fonte: elaborado pelo autor

## 3.2.2.1 Preparação e engenharia de variáveis

Diferentemente do ARIMA, cuja entrada é a própria série temporal univariada, o XGBoost exige que a série seja transformada em uma base tabular. Serão criadas variáveis defasadas, médias móveis e estatísticas que descrevam a série ao longo do tempo. Além disso, poderão ser adicionadas  variáveis  de  calendário  (mês,  dia  da semana, feriados  etc.),  enriquecendo  o  conjunto  de  treinamento  com  informações

contextuais.  Esta  etapa  é  exclusiva  e  essencial  para  o  XGBoost,  pois  permite  ao modelo explorar dependências temporais e efeitos sazonais/exógenos.

## 3.2.2.2 Divisão dos dados em treino e teste

Assim como no ARIMA, os dados serão divididos em conjuntos de treino e teste, sempre respeitando a ordem cronológica para evitar vazamento de informações futuras.

## 3.2.2.3 Normalização e tratamento dos dados

Esta etapa, embora similar à limpeza realizada no ARIMA, será orientada para o  contexto  tabular.  Serão  tratados  valores  ausentes  gerados  na  criação  de  lags  e médias  móveis  por  meio  de  imputação  ou  exclusão.  Se  necessário,  as  variáveis poderão  ser  normalizadas  ou  padronizadas  para  garantir  melhor  desempenho  do algoritmo.

## 3.2.2.4 Configuração dos hiper parâmetros iniciais

Diferentemente do ARIMA, em que os parâmetros de configuração são (p, d, q) definidos com base em análise de autocorrelação da própria série temporal, o modelo XGBoost depende de um conjunto mais amplo de hiper parâmetros que controlam tanto a complexidade quanto o desempenho do algoritmo de árvores de decisão.

Entre os principais hiper parâmetros que deverão ser configurados inicialmente, destacam-se:

- a)  n\_estimators (número de árvores): Define quantas árvores de decisão serão criadas e combinadas pelo modelo.
- b)  max\_depth  (profundidade  máxima):  Limita  a  quantidade  de  divisões  que cada  árvore  pode  fazer,  afetando  a  capacidade  de  capturar  padrões complexos sem sobre ajuste.
- c)  learning\_rate (taxa de aprendizado): Controla o peso de cada nova árvore adicionada no processo de boosting, influenciando diretamente a velocidade e a estabilidade do treinamento.

- d)  subsample (amostragem): Determina a fração de exemplos utilizados para treinar cada árvore, o que pode aumentar a generalização do modelo.
- e)  colsample\_bytree: Define a proporção de variáveis consideradas em cada divisão, reduzindo a chance de sobre ajuste.

A seleção inicial desses hiper parâmetros poderão ser realizadas com base em estudos  prévios,  valores  sugeridos  na  literatura  ou  ainda  com  valores  padrão  do próprio XGBoost. É importante salientar que, diferentemente do ARIMA, o XGBoost permite  grande  flexibilidade  na  escolha  e  combinação  desses  hiper  parâmetros, tornando o processo de ajuste potencialmente mais complexo e exigente em termos de experimentação.

## 3.2.2.5 Treinamento inicial do modelo

O processo de treinamento inicial do XGBoost se diferencia substancialmente do ARIMA,  principalmente pela estrutura dos dados e pelo  mecanismo  de aprendizado.

Enquanto  o  ARIMA  utiliza  uma  série  temporal  univariada  e  ajusta  seus parâmetros para capturar padrões autorregressivos e de média móvel, o XGBoost irá trabalhar sobre uma base tabular composta por múltiplas features, incluindo variáveis defasadas (lags), médias móveis, variáveis sazonais e de calendário, entre outras. O modelo será treinado utilizando o conjunto de treino previamente definido, buscando construir  sucessivas  árvores  de  decisão  (de  acordo  com  o  número  definido  em n\_estimators ) que, em conjunto, minimizarão o erro de previsão.

Durante esse processo, cada nova árvore será construída para corrigir os erros cometidos pelas árvores anteriores, em um procedimento iterativo chamado boosting. O ajuste do modelo será realizado até que todos os dados de treino tenham sido utilizados para aprender os padrões relevantes da série temporal e de suas variáveis derivadas.

Ao  final  do  treinamento  inicial,  o  modelo  estará  preparado  para  realizar previsões sobre o conjunto de teste, e os resultados obtidos servirão como base para a avaliação inicial de desempenho e para eventuais ajustes de hiper parâmetros em etapas subsequentes.

## 3.2.2.6 Avaliação inicial de desempenho

A  avaliação  do  desempenho  inicial  será  realizada  de  maneira  análoga  ao ARIMA, por meio de métricas como RMSE, MAE ou MAPE, aplicadas ao conjunto de teste. A análise dos erros também poderá indicar a necessidade de ajuste nas features ou nos hiper parâmetros.

## 3.2.2.7 Busca e ajuste de hiper parâmetros

Enquanto o ajuste de parâmetros do ARIMA envolve os valores de p, d, q, no XGBoost será  realizada  uma  busca  sistemática  para  identificar  os  melhores  hiper parâmetros do modelo, como taxa de aprendizado, número de árvores e profundidade máxima.

## 3.2.2.8 Validação cruzada e análise de resultados

Assim como no ARIMA, será empregada validação cruzada adequada a séries temporais, assegurando a robustez dos resultados e a ausência de sobre ajuste. Os resultados da validação serão analisados quanto à consistência e possíveis padrões residuais.

## 3.2.2.9 Geração das previsões finais e armazenamento dos resultados

Por fim, as previsões finais geradas pelo modelo XGBoost serão armazenadas para comparação direta com os resultados do ARIMA, dos demais modelos avaliados e com as previsões atualmente geradas pelo Power BI.

## 3.2.3 Suavização exponencial

A figura 4 mostra a metodologia utilizada para o modelo.

Figura 4 - Metodologia do modelo de Suavização Exponencial

<!-- image -->

Fonte: elaborado pelo autor

## 3.2.3.1 Preparação dos dados

A preparação dos dados para o modelo de Suavização Exponencial seguirá o padrão estabelecido para os modelos estatísticos (ARIMA e Theta), sendo utilizada a série  temporal  original  em  formato  univariado  e  na  granularidade  apropriada  ao problema. Será garantida a ordenação cronológica, bem como a anonimização das informações conforme as diretrizes éticas. Dados ausentes serão tratados previamente, por meio de interpolação, imputação por média ou exclusão.

## 3.2.3.2 Análise exploratória e estrutura da série temporal

Será  conduzida  uma  análise  gráfica  e  estatística  para  identificação  de tendências, ciclos e possíveis padrões sazonais. Essa etapa será fundamental para determinar o tipo de suavização exponencial a ser adotado (simples, com tendência ou com tendência e sazonalidade), e para subsidiar a configuração dos parâmetros do modelo.

## 3.2.3.3 Divisão em conjunto de treino e teste

A base de dados será segmentada em conjuntos de treino e teste, preservando a sequência temporal dos registros. A proporção adotada poderá variar conforme o tamanho da amostra, mas a divisão seguirá a mesma lógica dos modelos estatísticos anteriores, visando simular o ambiente real de previsão.

## 3.2.3.4 Seleção do tipo de suavização exponencial e parâmetros

A  seleção  dos  parâmetros  do  modelo  será  realizada  com  base  na  análise exploratória e nas opções oferecidas pela implementação da biblioteca. Poderão ser especificados manualmente, ou deixados para ajuste automático, parâmetros como:

- a)  Tipo de tendência: aditiva, multiplicativa ou ausente;
- b)  Tipo de sazonalidade: aditiva, multiplicativa ou ausente;
- c)  Periodicidade sazonal, de acordo com a frequência da série (por exemplo, 12 para dados mensais com sazonalidade anual);
- d)  Uso ou não de tendência amortecida.

Caso a estrutura da série não seja evidente, a configuração automática dos parâmetros  será  empregada,  permitindo  ao  algoritmo  determinar  os  componentes mais adequados.

## 3.2.3.5 Treinamento inicial do modelo

O treinamento será realizado por meio do ajuste do modelo da biblioteca Darts ao  conjunto  de  treino,  utilizando  os  parâmetros  selecionados  na  etapa  anterior.  O algoritmo irá otimizar os coeficientes de suavização (como alfa, beta e gama) para minimizar  o  erro  de  previsão,  podendo  realizar  busca  automática  das  melhores configurações.  Ressalta-se  que  a  série  de  entrada  deverá  estar  livre  de  valores ausentes e ser estritamente univariada, conforme as exigências do modelo.

## 3.2.3.6 Geração das previsões

Após o treinamento, serão geradas as previsões para o horizonte de teste. As previsões serão extraídas a partir do modelo ajustado, permitindo a comparação direta com os valores reais observados.

## 3.2.3.7 Avaliação do desempenho

O desempenho do modelo de Suavização Exponencial será aferido por meio de  métricas  como  RMSE,  MAE  ou  MAPE,  as  mesmas  empregadas  nos  demais modelos.  Essa  abordagem  garantirá  a  comparabilidade  entre  todos  os  métodos avaliados.

## 3.2.3.8 Ajuste fino e revalidação

Se necessário, serão realizados ajustes nos parâmetros do modelo, como a seleção de periodicidade sazonal diferente ou a alteração da estrutura de tendência.

Novos treinamentos e avaliações serão conduzidos até que se atinja um desempenho considerado robusto.

## 3.2.3.9 Geração das previsões finais e armazenamento dos resultados

Os  resultados,  incluindo  previsões,  resíduos  e  parâmetros  utilizados,  serão armazenados  de  maneira  estruturada.  Esses  resultados  serão  então  comparados futuramente com os demais modelos implementados e com a previsão atualmente utilizada no Power BI, visando identificar a abordagem mais precisa e confiável.

## 3.2.4 Theta

A figura 5 mostra a metodologia utilizada para o modelo.

Figura 5 - Metodologia do modelo Theta

<!-- image -->

Fonte: elaborado pelo autor

## 3.2.4.1 Organização e pré-condições dos dados

Antes de qualquer processamento, a série temporal será conferida quanto à univariança e ausência de valores nulos, pois estas são condições  indispensáveis para a aplicação do Theta na Darts. Caso sejam identificados dados ausentes, serão realizados  procedimentos  de  interpolação  ou  eliminação  dos  registros  afetados.  A granularidade  e  ordenação  cronológica  também  serão  revisadas,  assegurando  a integridade sequencial da série.

## 3.2.4.2 Análise inicial e sazonalidade

A análise exploratória, com foco em tendências e padrões repetitivos, orientará a parametrização do modelo. Se a sazonalidade for uma característica relevante da série,  já  identificada  em  etapas  anteriores  ou  confirmada  aqui,  alguns  parâmetros serão definidos explicitamente, assim como o período sazonal. Em casos de incerteza quanto  à  presença  de  sazonalidade,  será  mantida  a  configuração  automática  do Theta.

## 3.2.4.3 Separação temporal para avaliação

A divisão entre dados para treino e teste respeitará a lógica já adotada ao longo do trabalho: os registros mais antigos comporão a base de aprendizagem do modelo, enquanto  o  trecho  final  da  série  será  reservado  exclusivamente  para  avaliação preditiva.  Essa  separação  garante  que  as  previsões  simulem  um  cenário  real  de atualização e monitoramento contínuo.

## 3.2.4.4 Configuração e execução do algoritmo

A etapa de configuração no Darts é simplificada pelo caráter automático do modelo Theta,  dispensando  a necessidade de  ajustes manuais  extensos.  Quando apropriado, serão explicitados parâmetros, priorizando reprodutibilidade e alinhamento com os padrões identificados na análise inicial. A execução do treinamento será realizada diretamente pela biblioteca, com o Theta operando sobre o conjunto de treino e processando internamente a decomposição e recomposição da série segundo sua abordagem matemática característica.

## 3.2.4.5 Produção das previsões e pós-processamento

Com  o  modelo  ajustado,  serão  produzidas  as  previsões  para  o  intervalo definido de teste. Os resultados, extraídos diretamente do modelo, serão posteriormente reintegrados ao fluxo de avaliação conjunta com os demais algoritmos aplicados.

## 3.2.4.6 Avaliação quantitativa e diagnóstico

O desempenho do Theta será analisado utilizando métricas padronizadas do projeto, permitindo não apenas comparar acurácia, mas também  avaliar o comportamento residual e identificar possíveis limitações do modelo frente a outliers ou mudanças estruturais da série.

## 3.2.4.7 Iteração e consolidação dos resultados

Na  hipótese  de  resultados  insatisfatórios,  o  fluxo  prevê  nova  análise  dos parâmetros  de  sazonalidade  e  repetição  do  ciclo  de  ajuste  e  teste.  As  melhores configurações e resultados obtidos serão devidamente documentados e os dados de previsões armazenados de forma compatível com os demais experimentos.

## 3.3 AVALIAÇÃO E COMPARAÇÃO DOS MODELOS

Após o ajuste e validação de todos os modelos preditivos considerados neste trabalho,  será  realizada  uma  comparação  quantitativa  do  desempenho  de  cada modelo utilizando  as  seguintes  métricas  estatísticas,  recomendadas  pela  literatura para problemas de previsão de séries temporais:

- a)  Erro Médio Absoluto (MAE);
- b)  Raiz do Erro Quadrático Médio (RMSE);
- c)  Erro Percentual Absoluto Médio (MAPE).

Essas métricas serão calculadas para o conjunto de teste de cada modelo. O modelo que apresentar o menor valor de erro (considerando principalmente MAE e RMSE),  será  selecionado  como  o  modelo  de  melhor  desempenho,  conforme abordagem utilizada por Hyndman et al. (1999) e Gardner (1985).

Na sequência, o modelo de melhor desempenho será comparado diretamente ao método de previsão atualmente empregado no Power BI. Essa comparação será realizada utilizando as mesmas métricas, com o objetivo de identificar se a abordagem baseada  em  aprendizado  de  máquina  ou  métodos  estatísticos  apresenta  ganhos significativos de acurácia em relação à solução já adotada no produto da empresa.

A  escolha  final  do  modelo  será  baseada  não  apenas  no  desempenho quantitativo,  mas  também  na  sua  viabilidade  de  implementação  e  integração  à plataforma existente, conforme recomendam Gardner (1985) e Hyndman et al. (1999).

## 3.4 CRONOGRAMA

Quadro 1 - Cronograma de Desenvolvimento do Projeto

| Etapas do Projeto                              | Agosto   | Setembro   | Outubro   | Novembro   | Dezembro   |
|------------------------------------------------|----------|------------|-----------|------------|------------|
| Coleta e Anonimização dos Dados                | X        |            |           |            |            |
| Pré-processamento e Estruturação dos Dados     | X        |            |           |            |            |
| Análise Exploratória e Visualizações           | X        |            |           |            |            |
| Implementação do Modelo ARIMA                  |          | X          |           |            |            |
| Implementação do Modelo Suavização Exponencial |          |            | X         |            |            |
| Implementação do Modelo Theta                  |          |            | X         |            |            |
| Implementação do Modelo XGBoost                |          |            |           | X          |            |
| Validação, Ajuste Fino e Seleção dos Modelos   |          |            |           | X          |            |
| Comparação com Power BI                        |          |            |           | X          |            |
| Documentação dos Resultados                    |          |            |           | X          | X          |

Fonte: elaborado pelo autor

## REFERÊNCIAS

ASSIMAKOPOULOS, V.; NIKOLOPOULOS, K. The theta model: a decomposition approach to forecasting. International Journal of Forecasting , v. 16, n. 4, p. 521 -530, out. 2000. Disponível em: https://doi.org/10.1016/S0169-2070(00)00066-2.

BEZERRA, Manoel Ivanildo Silvestre. Apostila de Análise de Séries Temporais . São Paulo: UNESP, 2006. Disponível em:

https://www.ibilce.unesp.br/Home/Departamentos/MatematicaEstatistica/apostila\_ser ies\_temporais\_unesp.pdf.

BOX, G. E. P. et al. Time series analysis: forecasting and control . Hoboken, New Jersey: John Wiley &amp; Sons, 2015.

CHEN, T.; GUESTRIN, C. XGBoost: a Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining KDD '16 , v. 1, n. 1, p. 785 -794, 13 ago. 2016. Disponível em: https://doi.org/10.1145/2939672.2939785.

DAIRU, X.; SHILONG, Z. Machine Learning Model for Sales Forecasting by Using XGBoost . Disponível em: https://doi.org/10.1109/ICCECE51280.2021.9342304.

ENSAFI, Y. et al. Time-series forecasting of seasonal items sales using machine learning -A comparative analysis. International Journal of Information Management Data Insights , v. 2, n. 1, p. 100058, abr. 2022. Disponível em: https://doi.org/10.1016/j.jjimei.2022.100058.

FATTAH, J. et al. Forecasting of demand using ARIMA model. International Journal of Engineering Business Management , v. 10, n. 1, p. 184797901880867, jan. 2018. Disponível em: https://journals.sagepub.com/doi/10.1177/1847979018808673.

FIORUCCI, J. A. et al. Models for optimising the theta method and their relationship to state space models. International Journal of Forecasting , v. 32, n. 4, p. 1151 -1161, out. 2016. Disponível em: https://doi.org/10.1016/j.ijforecast.2016.02.005.

FOURKIOTIS, K. P.; TSADIRAS, A. Applying Machine Learning and Statistical Forecasting Methods for Enhancing Pharmaceutical Sales Predictions. Forecasting , v. 6, n. 1, p. 170 -186, 1 mar. 2024. Disponível em:

https://doi.org/10.3390/forecast6010010.

GARDNER, E. S. Exponential smoothing: The state of the art. Journal of Forecasting , v. 4, n. 1, p. 1 -28, 1985. Disponível em: https://doi.org/10.1002/for.3980040103.

KONTOPOULOU, V. I. et al. A Review of ARIMA vs. Machine Learning Approaches for Time Series Forecasting in Data Driven Networks. Future Internet , v. 15, n. 8, p. 255, 1 ago. 2023. Disponível em: https://doi.org/10.3390/fi15080255.

LOZIA, Z. Application of modelling and simulation to evaluate the theta method used in diagnostics of automotive shock absorbers. The Archives of Automotive Engineering -Archiwum Motoryzacji , v. 96, n. 2, p. 5 -30, 30 jun. 2022. Disponível em: https://doi.org/10.14669/AM/150823.

MAKRIDAKIS, S.; HIBON, M. The M3-Competition: results, conclusions and implications. International Journal of Forecasting , v. 16, n. 4, p. 451 -476, out. 2000. Disponível em: https://doi.org/10.1016/S0169-2070(00)00057-1.

MAKRIDAKIS, S.; WHEELWRIGHT, S. C.; HYNDMAN, R. J. Forecasting: Methods and Applications. In: Elements of Forecasting . Oxfordshire: Taylor &amp; Francis, 1999. p. 345 -346. Disponível em:

https://www.researchgate.net/publication/52008212\_Forecasting\_Methods\_and\_Appl ications.

MALIK, Shubham; HARODE, Rohan; KUNWAR, Akash Singh. XGBoost: a deep dive into boosting . Medium Blog, 2020. Disponível em: http://dx.doi.org/10.13140/RG.2.2.15243.64803.

MCKENZIE, ED. General exponential smoothing and the equivalent arma process. Journal of Forecasting , v. 3, n. 3, p. 333 -344, jul. 1984. Disponível em: https://doi.org/10.1002/for.3980030312.

MONDAL, P.; SHIT, L.; GOSWAMI, S. Study of Effectiveness of Time Series Modeling (Arima) in Forecasting Stock Prices. International Journal of Computer Science, Engineering and Applications , v. 4, n. 2, p. 13 -29, 30 abr. 2014. Disponível em: https://doi.org/10.5121/ijcsea.2014.4202.

MURAT, M. et al. Forecasting daily meteorological time series using ARIMA and regression models. International Agrophysics , v. 32, n. 2, p. 253 -264, 1 abr. 2018. Disponível em: https://doi.org/10.1515/intag-2017-0007.

NEWBOLD, P. ARIMA model building and the time series analysis approach to forecasting. Journal of Forecasting , v. 2, n. 1, p. 23 -35, jan. 1983. Disponível em: https://doi.org/10.1002/for.3980020104.

PAO, James J.; SULLIVAN, Danielle S. Time series sales forecasting . Final year project, Computer Science, Stanford Univ., Stanford, CA, USA, 2017. Disponível em: https://cs229.stanford.edu/proj2017/final-reports/5244336.pdf.

The Annals of Mathematical

PARZEN, E. An Approach to Time Series Analysis. Statistics , v. 32, n. 4, p. 951 -989, 1961. Disponível em: https://www.jstor.org/stable/2237900.

SHIRI, F. M. et al. A Comprehensive Overview and Comparative Analysis on Deep Learning Models. Journal on Artificial Intelligence , v. 6, n. 1, p. 301 -360, 2024. Disponível em: https://doi.org/10.32604/jai.2024.054314.

SPILIOTIS, E.; ASSIMAKOPOULOS, V.; MAKRIDAKIS, S. Generalizing the Theta method for automatic forecasting. European Journal of Operational Research , jan. 2020. Disponível em: http://dx.doi.org/10.1016/j.ejor.2020.01.007.

VAVLIAKIS, K.; SIAILIS, A.; SYMEONIDIS, A. Optimizing Sales Forecasting in eCommerce with ARIMA and LSTM Models. Proceedings of the 17th International Conference on Web Information Systems and Technologies , 2021. Disponível em: https://doi.org/10.5220/0010659500003058.