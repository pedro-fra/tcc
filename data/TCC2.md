## UNIVERSIDADE DO VALE DO RIO DOS SINOS (UNISINOS)

## UNIDADE ACADÊMICA DE GRADUAÇÃO CURSO DE ENGENHARIA DA

## COMPUTAÇÃO

## PEDRO DELAVALD FRÁ

## PREVISÃO DE VENDAS:

Análise comparativa entre abordagens de aprendizado de máquina e Power BI
São Leopoldo
2025

---

# Page 2

2

## PEDRO DELAVALD FRÁ

## PREVISÃO DE VENDAS:

Análise comparativa entre abordagens de aprendizado de máquina e Power BI
Trabalho de
Conclusão de
Curso apresentado como requisito parcial para
obtenção do título de Bacharel em
Engenharia da Computação, pelo Curso de
Engenharia da
Computação da
Universidade do Vale do Rio dos Sinos

## (UNISINOS)

Orientador: Prof. MSc. Jean Schmith
São Leopoldo
2025

---

# Page 3

## RESUMO

Este trabalho tem como objetivo avaliar e comparar o desempenho de diferentes métodos de previsão de vendas, utilizando tanto técnicas estatísticas
tradicionais quanto algoritmos modernos de aprendizado de máquina, aplicados a dados reais de faturamento extraídos de um dashboard corporativo em Power BI.
Diante do aumento da competitividade e da demanda por decisões empresariais baseadas em dados, destaca-se a necessidade de modelos preditivos cada vez mais
precisos e robustos. O estudo envolve a implementação dos modelos ARIMA, Theta, Suavização Exponencial e XGBoost, analisando suas performances preditivas e as
possibilidades de adoção dessas abordagens no contexto empresarial. Os resultados são avaliados a partir de métricas estatísticas padronizadas, permitindo identificar se
algum modelo apresenta desempenho superior ao método atualmente empregado. A pesquisa contribui para a aproximação entre teoria e prática, oferecendo subsídios
para a escolha de métodos de previsão mais adequados às necessidades das organizações e potencializando o valor estratégico das análises de vendas.
Palavras-chave: Previsão de Vendas; Séries Temporais; Aprendizado de Máquina;
Power BI; ARIMA; XGBoost; Suavização Exponencial; Método Theta; Business
Intelligence.

---

# Page 4

## ABSTRACT

This work aims to evaluate and compare the performance of different sales forecasting methods, employing both traditional statistical techniques and modern
machine learning algorithms, applied to real revenue data extracted from a corporate dashboard in Power BI. Given the increasing competitiveness and demand for data-
driven business decisions, there is a growing need for more accurate and robust predictive models. The study involves the implementation of ARIMA, Theta,
Exponential Smoothing, and XGBoost models, analyzing their predictive performance and the feasibility of adopting these approaches in corporate environments. The results
are assessed using standardized statistical metrics, allowing for the identification of models that outperform the currently employed method. This research contributes to
bridging the gap between theory and practice, offering guidance for the selection of forecasting methods that best fit organizational needs and enhancing the strategic
value of sales analytics.
Key-words: Sales Forecasting; Time Séries; Machine Learning; Power BI; ARIMA;
XGBoost; Exponential Smoothing; Theta Method; Business Intelligence.

---

# Page 5

## LISTA DE FIGURAS

Figura 1 - Metodologia geral do trabalho ................................................................... 30
Figura 2 - Metodologia do pré-processamento .......................................................... 32
Figura 3 - Visão geral da série temporal ................................................................... 36
Figura 4 – Decomposição da série temporal ............................................................. 37
Figura 5 - Análise da sazonalidade ........................................................................... 38
Figura 6 - Propriedades estatísticas da série temporal ............................................. 39
Figura 7 - Análise de distribuição .............................................................................. 40
Figura 8 - Evolução temporal das vendas ................................................................. 41
Figura 9 - Análise de correlação temporal ................................................................. 42
Figura 10 - Metodologia do modelo ARIMA .............................................................. 44
Figura 11 – Metodologia do modelo Suavização Exponencial .................................. 51
Figura 12 – Metodologia do modelo Theta ................................................................ 55
Figura 13 – Metodologia do modelo XGBoost ........................................................... 58

---

# Page 6

6

## LISTA DE QUADROS

Nenhuma entrada de índice de ilustrações foi encontrada.

---

# Page 7

## LISTA DE SIGLAS

## CNN

Convolutional Neural Network

## RNN

Recurrent Neural Network

## ARIMA

Auto Regressive Integrated Moving Average
XGBoost
X Gradient Boost

## ML

Machine Learning

## PIB

Produto Interno Bruto

## SARIMA

Seasonal Auto Regressive Integrated Moving Average

## LSTM

Long Short-Term Memory

## STL

Seasonal and Trend decomposition using LOESS

## AIC

Akaike Information Criterion

## AR

Auto Regressive

## MA

Moving Average

## SES

Simple Exponential Smoothing

## ACF

Autocorrelation Function

## PACF

Parcial Autocorrelation Function

## KPSS

Kwiatkowski-Phillips-Schmidt-Shin

## ADF

Augmented Dickey Fuller

## RMSSE

Root Mean Squared Scaled Error

## RMSE

Root Mean Squared Error

## MAE

Mean Absolute Error

## BI

Business Intelligence

## GBDT

Gradient Boosting Decision Tree

---

# Page 8

## SUMÁRIO

1 INTRODUÇÃO ....................................................................................................... 11
1.1 TEMA .................................................................................................................. 11
1.2 DELIMITAÇÃO DO TEMA ................................................................................... 12
1.3 PROBLEMA ........................................................................................................ 12
1.4 OBJETIVOS ........................................................................................................ 12
1.4.1 Objetivo geral ................................................................................................. 12
1.4.2 Objetivos específicos ..................................................................................... 12
1.5 JUSTIFICATIVA .................................................................................................. 13
2 FUNDAMENTAÇÃO TEÓRICA ............................................................................. 14
2.1 SÉRIES TEMPORAIS ......................................................................................... 14
2.1.1 Conceitos fundamentais e definições .......................................................... 14
2.1.2 Características principais .............................................................................. 14
2.1.3 Classificações de séries temporais .............................................................. 15
2.1.4 Exemplos de aplicação .................................................................................. 16
2.2 MÉTODO THETA ................................................................................................ 16
2.2.1 Descrição geral e origem ............................................................................... 17
2.2.2 Fundamentação teórica e parâmetros .......................................................... 17
2.2.3 Equação da linha Theta ................................................................................. 18
2.2.4 Expressões aditivas e multiplicativas .......................................................... 18
2.2.5 Funcionamento do método para previsão de dados futuros ..................... 18
2.2.6 Exemplos práticos de uso ............................................................................. 19
2.3 MODELO ARIMA ................................................................................................ 20
2.3.1 Definição e estrutura do modelo ARIMA ...................................................... 20
2.3.2 Conceitos e características do modelo ARIMA ........................................... 21
2.3.3 Como o modelo ARIMA funciona para prever dados futuros? .................. 21
2.3.4 Casos práticos e exemplos na literatura ...................................................... 22
2.4 SUAVIZAÇÃO EXPONENCIAL ........................................................................... 23
2.4.1 Definição e estrutura do método .................................................................. 23
2.4.2 Vantagens e limitações na previsão de dados ............................................ 24
2.4.3 Aplicações e estudos de caso ...................................................................... 25
2.5 XGBOOST ........................................................................................................... 26
2.5.1 Visão geral do Extreme Gradient Boosting .................................................. 26

---

# Page 9

9
2.5.2 Características e conceitos do XGBoost ..................................................... 27
2.5.3 Como o XGBoost prevê dados futuros ........................................................ 27
2.5.4 Exemplos práticos de uso do XGBoost ........................................................ 29
3 METODOLOGIA .................................................................................................... 30
3.1 METODOLOGIA DE TRABALHO ....................................................................... 30
3.1.1 Definição do problema e objetivos da previsão .......................................... 31
3.1.2 Coleta e pré-processamento dos dados ...................................................... 31
3.1.2.1 Filtragem e agregação inicial ......................................................................... 32
3.1.2.3 Agregação temporal mensal .......................................................................... 33
3.1.2.4 Conversão para formato Darts ...................................................................... 33
3.1.2.5 Considerações sobre engenharia de features ............................................... 33
3.1.3 Análise exploratória e estruturação da série temporal ............................... 34
3.1.3.1 Visão geral da série temporal ........................................................................ 35
3.1.3.2 Decomposição STL ....................................................................................... 36
Fonte: elaborado pelo autor ...................................................................................... 37
3.1.3.3 Análise de sazonalidade ................................................................................ 37
3.1.3.4 Propriedades estatísticas .............................................................................. 38
Fonte: elaborado pelo autor ...................................................................................... 39
3.1.3.5 Análise de distribuição ................................................................................... 39
3.1.3.6 Evolução temporal detalhada ........................................................................ 40
Fonte: elaborado pelo autor ...................................................................................... 41
3.1.3.7 Análise de correlação temporal ..................................................................... 41
Fonte: elaborado pelo autor ...................................................................................... 42
3.1.3.8 Insights para modelagem .............................................................................. 42
3.2 MODELOS DE PREVISÃO UTILIZADOS ........................................................... 43
3.2.1 ARIMA .............................................................................................................. 44
3.2.1.1 Importação das bibliotecas e configuração do ambiente ............................... 44
3.2.1.2 Ingestão e conversão dos dados para série temporal ................................... 45
3.2.1.3 Verificação de estacionaridade e diferenciação ............................................ 46
3.2.1.4 Divisão dos dados em conjuntos de treino e teste ........................................ 46
3.2.1.5 Definição dos parâmetros p, d e q ................................................................. 47
3.2.1.6 Treinamento do modelo ................................................................................. 48
3.2.1.7 Validação do modelo e ajustes finos ............................................................. 49
3.2.1.8 Análise residual ............................................................................................. 50

---

# Page 10

10
3.2.1.9 Armazenamento dos resultados para comparação futura ............................. 50
3.2.2 Suavização Exponencial ................................................................................ 51
3.2.2.1 Análise de componentes para seleção do modelo ........................................ 52
3.2.2.2 Decisão entre modelo aditivo e multiplicativo ................................................ 52
3.2.2.3 Configuração e otimização de parâmetros .................................................... 53
3.2.2.4 Treinamento por suavização recursiva .......................................................... 53
3.2.2.5 Geração de previsões diretas ........................................................................ 54
3.2.2.6 Análise residual específica para suavização ................................................. 54
3.2.3 Theta ................................................................................................................ 54
3.2.3.1 Verificação de pré-condições do método Theta ............................................ 55
3.2.3.2 Configuração automática vs. manual do modelo ........................................... 56
3.2.3.3 Decomposição e criação das linhas Theta .................................................... 56
3.2.3.4 Treinamento e ajuste das componentes ........................................................ 57
3.2.3.5 Combinação de previsões e extrapolação ..................................................... 57
3.2.3.6 Avaliação e diagnósticos específicos ............................................................ 57
3.2.4 XGBoost .......................................................................................................... 58
3.2.4.1 Preparação e integração com Darts .............................................................. 58
3.2.4.2 Divisão dos dados em treino e teste Engenharia automática de features ..... 59
3.2.4.3 Engenharia automática de features ............................................................... 59
3.2.4.4 Configuração dos hiper parâmetros iniciais ................................................... 60
3.2.4.5 Treinamento do modelo ................................................................................. 61
3.2.4.6 Avaliação inicial de desempenho .................................................................. 62
3.2.4.7 Validação e análise de resultados ................................................................. 62
3.2.4.8 Geração das previsões finais e armazenamento dos resultados .................. 62
3.3 AVALIAÇÃO E COMPARAÇÃO DOS MODELOS .............................................. 62
REFERÊNCIAS ......................................................................................................... 63

---

# Page 11

11

## 1 INTRODUÇÃO

A previsão de vendas, no contexto atual da transformação digital e da crescente demanda por decisões empresariais baseadas em dados, se estabelece como um dos
grandes desafios e diferenciais competitivos para organizações de todos os portes.
Com mercados cada vez mais dinâmicos e suscetíveis a variações econômicas, tecnológicas e comportamentais, a precisão nas estimativas de faturamento assume
papel central no planejamento, controle de estoques, logística, definição de metas e estratégias comerciais. Este cenário impulsionou o avanço de diferentes métodos de
previsão, desde técnicas estatísticas tradicionais até abordagens inovadoras de aprendizado de máquina, que vêm transformando a forma como as empresas
analisam e projetam seus resultados futuros.
O uso disseminado de ferramentas de BI, como o Power BI, trouxe grandes avanços para a visualização e interpretação dos dados históricos das empresas,
permitindo a elaboração de dashboards customizados para acompanhamento do desempenho de vendas. Contudo, muitos desses sistemas ainda utilizam métodos de
previsão relativamente simples, que podem não captar integralmente a complexidade dos padrões temporais, sazonalidades e variáveis exógenas presentes nos dados
(ENSAFI et al., 2022). Paralelamente, algoritmos de ML, como o XGBoost, vêm sendo destacados na literatura por sua elevada acurácia preditiva, robustez e flexibilidade
na incorporação de múltiplos fatores ao processo de modelagem, sendo escolhido frequentemente em cenários reais e competições internacionais (CHEN; GUESTRIN,
2016).
Diante desse contexto, torna-se pertinente avaliar, sob uma perspectiva aplicada e comparativa, se modelos de ML podem efetivamente aprimorar as
previsões de faturamento realizadas por soluções já consolidadas no ambiente empresarial, como o Power BI, contribuindo para a geração de insights mais robustos
e embasados para a tomada de decisão.

## 1.1 TEMA

O presente trabalho aborda o tema da previsão de vendas utilizando séries temporais, com foco na comparação entre métodos tradicionais e modernos de
modelagem preditiva aplicados a dados reais de faturamento empresarial.

---

# Page 12

12

## 1.2 DELIMITAÇÃO DO TEMA

A pesquisa concentra-se na análise comparativa do desempenho de diferentes modelos de previsão utilizando dados históricos extraídos de um banco de dados. O
estudo limita-se à previsão de faturamento mensal, simulando o contexto prático enfrentado por empresas que necessitam estimar o resultado do mês corrente com
base em informações parciais, do primeiro dia do mês até o momento da consulta.

## 1.3 PROBLEMA

O problema que orienta este trabalho é: Modelos avançados de aprendizado de máquina podem proporcionar previsões mais precisas de faturamento, quando
comparados à abordagem utilizada em dashboards de Power BI? A investigação busca responder se a adoção de modelos de aprendizado de máquina como XGBoost,
ARIMA, Suavização Exponencial e Theta pode, de fato, melhorar a acurácia das projeções realizadas atualmente pela empresa, promovendo maior confiabilidade e
valor estratégico às informações disponibilizadas.

## 1.4 OBJETIVOS

1.4.1 Objetivo geral
Avaliar, de forma comparativa, o desempenho de diferentes abordagens de previsão de vendas, sejam elas tradicionais ou baseadas em ML, aplicadas a dados
reais de faturamento, verificando se algum dos modelos apresenta desempenho superior ao método atualmente utilizado em dashboards de Power BI.
1.4.2 Objetivos específicos a) Revisar e contextualizar os principais conceitos de séries temporais,
métodos estatísticos clássicos e técnicas de ML voltadas à previsão de vendas, conforme descrito por autores como Bezerra (2006), Makridakis,
Wheelwright e Hyndman (1999) e Ensafi et al. (2022);

---

# Page 13

13 b) Estruturar e pré-processar os dados históricos de faturamento de acordo
com as exigências de cada modelo preditivo, assegurando anonimização, integridade e conformidade com boas práticas de ciência de dados;
c) Implementar, treinar e validar modelos de previsão ARIMA, Theta, Suavização Exponencial e XGBoost, utilizando métricas estatísticas
padronizadas para avaliação do desempenho;
d) Analisar comparativamente os resultados obtidos e discutir as vantagens, limitações e possibilidades práticas para adoção dos métodos preditivos no
contexto empresarial.
Acredita-se que essa abordagem possibilite uma análise abrangente e rigorosa, identificando as oportunidades e desafios envolvidos na transição para modelos
preditivos mais avançados no ambiente corporativo.

## 1.5 JUSTIFICATIVA

A relevância deste estudo se justifica tanto pelo avanço recente das técnicas de análise preditiva quanto pela necessidade real de organizações aprimorarem seus
processos de tomada de decisão frente a cenários de incerteza e competitividade. Do ponto de vista acadêmico, há uma lacuna na literatura nacional sobre aplicações
práticas e comparativas de modelos de machine learning em ambientes de BI amplamente adotados por empresas brasileiras, como o Power BI (ENSAFI et al.,
2022; SHIRI et al., 2024). Internacionalmente, pesquisas vêm demonstrando o potencial de algoritmos como XGBoost na superação de métodos tradicionais de
previsão, especialmente em séries temporais com padrões complexos e influências externas (CHEN; GUESTRIN, 2016).
No âmbito empresarial, a adoção de modelos mais precisos pode representar ganhos substanciais em planejamento, controle financeiro e competitividade,
permitindo que decisões sejam tomadas com maior base quantitativa e menor risco.
Este trabalho, ao propor uma análise comparativa fundamentada, contribui para aproximar a teoria e a prática, orientando gestores e profissionais de dados quanto à
melhor escolha de métodos para suas demandas específicas.

---

# Page 14

14

## 2 FUNDAMENTAÇÃO TEÓRICA

Neste capítulo, apresenta-se o embasamento teórico indispensável ao desenvolvimento do presente estudo. Serão discutidos os conceitos fundamentais
relacionados à previsão de dados, contemplando tanto a aplicação de algoritmos de aprendizado de máquina quanto a utilização de cálculos no Power BI. A partir dessa
fundamentação, busca-se sustentar o estudo de caso realizado, evidenciando as principais vantagens e limitações de cada abordagem na análise e projeção de
informações.

## 2.1 SÉRIES TEMPORAIS

A análise de séries temporais é uma importante área da estatística, dedicada à compreensão, modelagem e previsão de fenômenos que são observados de forma
sequencial no tempo. Conforme Bezerra (2006), a utilização da análise de séries temporais é amplamente difundida em diversas áreas, como economia, meteorologia,
saúde, controle de processos industriais, vendas e finanças, devido à capacidade de identificar padrões de comportamento e realizar previsões futuras com base em dados
históricos.
2.1.1 Conceitos fundamentais e definições
De acordo com Parzen (1961), uma série temporal pode ser entendida como um conjunto de observações dispostas cronologicamente, sendo representada
matematicamente como um processo estocástico, no qual cada valor observado corresponde a um instante específico no tempo.
2.1.2 Características principais
Entre os principais conceitos e características envolvidos na análise de séries temporais, destacam-se:
a) Estacionariedade: Segundo Bezerra (2006), a estacionariedade ocorre quando as propriedades estatísticas, tais como média, variância e
covariância, permanecem constantes ao longo do tempo. A condição de

---

# Page 15

15 estacionariedade é importante para aplicação correta de diversos modelos,
como os modelos ARIMA;
b) Tendência: Refere-se à direção predominante da série ao longo do tempo, podendo ser crescente, decrescente ou estável. Segundo Makridakis,
Wheelwright e Hyndman (1999), a tendência é fundamental para entender o comportamento das séries e escolher modelos adequados;
c) Sazonalidade: Corresponde às variações periódicas e regulares que ocorrem em intervalos fixos, como mensal ou anual, devido a fatores
externos ou
eventos recorrentes

## (MAKRIDAKIS,

## WHEELWRIGHT;

## HYNDMAN, 1999);

d) Autocorrelação: Representa a correlação da série consigo mesma em diferentes momentos do tempo (lags). De acordo com Parzen (1961), esse
conceito é fundamental para identificar e compreender o comportamento das séries temporais;
e) Ruído branco: Para Bezerra (2006), é a parcela aleatória da série temporal, composta por erros aleatórios independentes com média zero e
variância constante, que não apresentam qualquer tipo de padrão previsível.
2.1.3 Classificações de séries temporais
Makridakis, Wheelwright e Hyndman (1999) classificam as séries temporais em tipos distintos:
a) Séries estacionárias: Caracterizam-se por apresentar média e variância constantes ao longo do tempo. São frequentemente observadas em séries
financeiras de retorno;
b) Séries não estacionárias: São séries cujas propriedades estatísticas, como média e/ou variância, alteram-se com o tempo. Exemplos comuns incluem
séries econômicas como PIB e inflação;
c) Séries lineares e não lineares: Séries lineares podem ser modeladas por técnicas tradicionais, como ARIMA, enquanto séries não lineares exigem
modelos mais avançados, como redes neurais artificiais (SHIRI et al., 2024).

---

# Page 16

16
2.1.4 Exemplos de aplicação
Vários estudos demonstram a aplicação prática das séries temporais em diversos contextos:
a) Previsão de vendas no varejo: Ensafi et al. (2022) compararam técnicas tradicionais como SARIMA e Suavização Exponencial com métodos
avançados como redes neurais LSTM e CNN para previsão das vendas sazonais de móveis. Os resultados mostraram que as redes neurais LSTM
apresentaram maior precisão na captura de padrões complexos e sazonais;
b) Previsão de vendas semanais em lojas de departamento: Pao e Sullivan
(2014) utilizaram técnicas como árvores de decisão, STL+ARIMA e redes neurais feed-forward com entradas temporais defasadas, concluindo que as
redes neurais tiveram um desempenho superior, capturando com eficiência as sazonalidades das vendas semanais;
c) Aplicação de Deep Learning em séries temporais complexas: Shiri et al.
(2024) realizaram uma revisão abrangente sobre o uso de modelos de deep learning, como CNN, RNN, LSTM e Transformer, em séries temporais. O
estudo apontou que técnicas modernas baseadas em deep learning têm se mostrado superiores às técnicas tradicionais, principalmente em séries
complexas e com grandes volumes de dados.

## 2.2 MÉTODO THETA

O método Theta ganhou popularidade ao vencer a competição M3 de previsões de séries temporais devido à sua simplicidade e eficiência em gerar previsões precisas
para diversos tipos de dados. Desde então, este método tem sido amplamente estudado e aprimorado, resultando em diferentes variantes que exploram seu
potencial para aplicações automáticas e mais robustas. (ASSIMAKOPOULOS;

## NIKILOPOULOS, 2000).

---

# Page 17

17
2.2.1 Descrição geral e origem
O método Theta é uma técnica de previsão uni variada que decompõe a série temporal original em componentes denominados "linhas Theta". Cada linha Theta é
obtida ajustando-se a curvatura dos dados originais através de um parâmetro θ aplicado às segundas diferenças da série original. (ASSIMAKOPOULOS;

## NIKILOPOULOS, 2000; SPILIOTIS; ASSIMAKOPOULOS; MAKRIDAKIS, 2020). A

combinação dessas linhas Theta gera previsões que equilibram tendências de curto e longo prazo. (ASSIMAKOPOULOS; NIKILOPOULOS, 2000).
2.2.2 Fundamentação teórica e parâmetros
As principais características do método Theta incluem:
a) Decomposição da série temporal: a série original é dividida em múltiplas linhas Theta, destacando diferentes características como tendências de
curto e longo prazo (ASSIMAKOPOULOS; NIKOLOPOULOS, 2000);
b) Parâmetro θ (Theta): controla a curvatura das linhas, com 𝜃< 1 enfatizando tendências de longo prazo e 𝜃> 1 destacando variações de curto prazo.

## (ASSIMAKOPOULOS;

## NIKOLOPOULOS,

2000;

## SPILIOTIS;

## ASSIMAKOPOULOS; MAKRIDAKIS, 2020);

c) Combinação de previsões: as previsões geradas a partir das linhas Theta são combinadas usando ponderações específicas para gerar resultados
mais robustos e precisos (FIORUCCI et al., 2016);
d) Flexibilidade e robustez: permite ajuste e adaptação automática dos parâmetros para diferentes séries temporais, tornando-o versátil para
diversos contextos (SPILIOTIS; ASSIMAKOPOULOS; MAKRIDAKIS, 2020);
e) Eficiência computacional: destaca-se pela simplicidade computacional, sendo fácil e rápido de implementar, especialmente quando comparado com
métodos mais complexos como ARIMA ou redes neurais (FIORUCCI et al., 2016);
f) Capacidade de generalização: é aplicável em séries temporais com diferentes padrões, como tendências lineares, não lineares, séries com

---

# Page 18

18 comportamento
sazonal e
séries irregulares

## (SPILIOTIS;

## ASSIMAKOPOULOS; MAKRIDAKIS, 2020);

g) Simplicidade na interpretação: oferece resultados facilmente interpretáveis, facilitando seu uso prático em ambientes corporativos e industriais
(FIORUCCI et al., 2016).
2.2.3 Equação da linha Theta
Segundo Spiliotis, Assimakopoulos e Makridakis (2020), o método Theta pode ser matematicamente descrito da seguinte forma:
Seja 𝑌𝑡 uma série temporal observada no tempo 𝑡. Uma linha Theta 𝑍𝑡(𝜃) é obtida pela expressão:
∇2𝑍𝑡(𝜃) = 𝜃∇2𝑌𝑡= 𝜃(𝑌𝑡−2𝑌(𝑡−1) + 𝑌(𝑡+2)), 𝑡= 3, … , 𝑛
onde ∇2𝑌𝑡 é o operador das segundas diferenças da série original 𝑌 no ponto 𝑡.
2.2.4 Expressões aditivas e multiplicativas
No método Theta, as previsões podem ser realizadas utilizando expressões aditivas ou multiplicativas:
a) Modelo aditivo: é o modelo original do método Theta, no qual as previsões são obtidas pela combinação linear aditiva das linhas Theta ajustadas

## (ASSIMAKOPOULOS; NIKOLOPOULOS, 2000);

b) Modelo multiplicativo: é uma extensão recente do método, permitindo modelar situações em que componentes como sazonalidade e tendência
interagem de forma multiplicativa, sendo especialmente útil em séries com tendência
exponencial ou
comportamento sazonal
multiplicativo

## (SPILIOTIS; ASSIMAKOPOULOS; MAKRIDAKIS, 2020).

2.2.5 Funcionamento do método para previsão de dados futuros
Para prever dados futuros, o método Theta realiza as seguintes etapas

## (ASSIMAKOPOULOS; NIKOLOPOULOS, 2000; FIORUCCI, 2016):

---

# Page 19

19 a) Decomposição: a série temporal é decomposta em linhas Theta com
diferentes curvaturas;
b) Extrapolação: cada linha é extrapolada individualmente, frequentemente usando métodos simples, como suavização exponencial simples (SES) para
tendências de curto prazo e regressão linear para tendências de longo prazo;
c) Combinação das linhas: as previsões individuais são combinadas, geralmente com pesos iguais ou otimizados, produzindo uma previsão final
robusta.
2.2.6 Exemplos práticos de uso
O método Theta tem sido amplamente aplicado em diversas áreas, demonstrando sua robustez:
a) Competição M3: a versão clássica do método Theta alcançou resultados superiores às demais técnicas na competição M3, uma famosa competição
internacional focada em métodos de previsão de séries temporais, especialmente em séries mensais e microeconômicas, destacando-se por
sua precisão e simplicidade (MAKRIDAKIS; HIBON, 2000);
b) Diagnóstico automotivo: Lozia (2022) utilizou o método Theta na avaliação diagnóstica de amortecedores automotivos, demonstrando a eficácia do
método em modelar e prever o comportamento dinâmico de sistemas mecânicos complexos;
c) Previsão automática: Spiliotis, Assimakopoulos e Makridakis (2020) propuseram generalizações do método Theta capazes de selecionar
automaticamente a forma mais apropriada (aditiva ou multiplicativa) e ajustar a inclinação das tendências, superando outros algoritmos automáticos em
competições recentes (como M4), especialmente em séries anuais.

---

# Page 20

20

## 2.3 MODELO ARIMA

O modelo ARIMA é uma técnica estatística amplamente utilizada para análise e previsão de séries temporais, desenvolvido por Box e Jenkins (1970). É
especialmente indicado para séries cujos valores passados e erros históricos podem ser utilizados para prever valores futuros (NEWBOLD, 1983).
2.3.1 Definição e estrutura do modelo ARIMA
O modelo ARIMA é uma combinação dos modelos autorregressivos (AR), integrados (I) e de médias móveis (MA), definidos pela seguinte notação geral ARIMA
(p, d, q), onde (NEWBOLD, 1983):
a) p: ordem do termo autorregressivo (AR), representa a relação linear entre a observação atual e as anteriores;
b) d: número de diferenciações necessárias para tornar a série estacionária;
c) q: ordem dos termos de média móvel (MA), que refletem os erros anteriores do modelo.
Matematicamente, o modelo ARIMA (p, d, q) pode ser expresso da seguinte forma (NEWBOLD, 1983):
𝑌𝑡=  𝛿 + 𝜙1𝑌𝑡−1 + 𝜙2𝑌𝑡−2 + … + 𝜙𝑝𝑌𝑡−𝑝− 𝜃1𝜀𝑡−1 − 𝜃2𝜀𝑡−2 − … − 𝜃𝑞𝜀𝑡−𝑞+ 𝜀𝑡
Onde:
•
𝑌𝑡: valor atual da série temporal.
•
𝑌𝑡−1, 𝑌𝑡−2,..., 𝑌𝑡−𝑝 : valores anteriores da série temporal (termos AR).
• 𝜀𝑡: erro aleatório (resíduos) com distribuição normal, média zero e variância
constante (ruído branco).
• 𝜀𝑡−1, 𝜀𝑡−2, ..., 𝜀𝑡−𝑞: erros anteriores da série (termos MA).
• 𝛿: constante.
• 𝜙1, 𝜙2, … , 𝜙𝑝: coeficientes do termo autorregressivo.
• 𝜃1, 𝜃2, … , 𝜃𝑞: coeficientes do termo de média móvel.

---

# Page 21

21
2.3.2 Conceitos e características do modelo ARIMA
As principais características do modelo ARIMA incluem (BOX; JENKINS, 1970;
FATTAH et al., 2018):
a) Flexibilidade: Pode ajustar-se a diversas séries temporais, incorporando tendência, ciclos e sazonalidade;
b) Necessidade de estacionariedade: Séries temporais precisam ser estacionárias para utilização correta do modelo. A estacionariedade é
geralmente obtida por diferenciação sucessiva das séries temporais;
c) Simplicidade: Fácil de compreender e implementar, apresentando resultados robustos em previsões de curto prazo.
Para verificar se uma série é estacionária, frequentemente são utilizados testes estatísticos como o teste Dickey-Fuller (ADF) e o teste KPSS (MURAT et al., 2018).
2.3.3 Como o modelo ARIMA funciona para prever dados futuros?
O processo de construção do modelo ARIMA segue a metodologia Box-
Jenkins, que possui as seguintes etapas (BOX; JENKINS, 1970; MONDAL et al., 2014):
a) Identificação do modelo: Determinação das ordens p, d e q, com base na análise gráfica das funções de autocorrelação (ACF) e autocorrelação
parcial (PACF);
b) Estimação dos parâmetros: Os coeficientes do modelo são estimados, normalmente utilizando o método da máxima verossimilhança;
c) Diagnóstico do modelo: Verificação da adequação do modelo por meio da análise dos resíduos (erros), usando testes como o teste de Ljung-Box e
critérios estatísticos como AIC (Critério de Informação de Akaike);
d) Previsão: Realização da previsão de valores futuros utilizando o modelo ajustado.

---

# Page 22

22
2.3.4 Casos práticos e exemplos na literatura
O modelo ARIMA tem diversas aplicações práticas, como evidenciado em diferentes estudos acadêmicos:
a) Previsão de demanda em indústrias alimentícias: Fattah et al. (2018) mostraram que o modelo ARIMA (1,0,1) foi eficaz em prever a demanda
futura, ajudando a empresa na gestão eficiente de estoques e redução de custos;
b) Previsão de vendas no e-commerce: Um modelo híbrido combinando
ARIMA com redes neurais LSTM foi utilizado para previsão precisa em ambientes com alta volatilidade, como o comércio eletrônico (VAVLIAKIS et
al., 2021);
c) Previsão no mercado farmacêutico: Fourkiotis e Tsadiras (2024) utilizaram
ARIMA em combinação com técnicas de aprendizado de máquina para prever a demanda por produtos farmacêuticos, mostrando sua eficácia em
capturar efeitos sazonais. Para enfrentar esse desafio, Fourkiotis e Tsadiras
(2024) utilizaram técnicas de análise uni variada de séries temporais para desenvolver previsões mais precisas. Os autores analisaram uma base de
dados real contendo 600.000 registros históricos de vendas provenientes de uma farmácia online, abrangendo um período entre 2014 e 2019. A
metodologia proposta envolveu as etapas de pré-processamento e limpeza de dados, segmentação dos dados, análise exploratória e identificação dos
padrões temporais, aplicação e comparação do modelo ARIMA com modelos avançados de ML como LSTM e XGBoost e, por fim, avaliação do
modelo com métricas específicas. Os resultados demonstraram que o modelo ARIMA apresentou uma boa capacidade preditiva ao capturar
adequadamente a sazonalidade e tendências lineares de vendas. Contudo, os autores destacaram que modelos de ML avançados, especialmente o
XGBoost, tiveram um desempenho ainda superior. Em particular, o XGBoost obteve as menores taxas de erro absoluto percentual médio (MAPE). Apesar
da boa performance dos modelos avançados de Machine Learning, o modelo ARIMA ainda obteve desempenho competitivo e foi considerado

---

# Page 23

23 eficaz especialmente em séries temporais com forte componente linear e
sazonalidade bem definida;
d) Previsão de preços no mercado financeiro: Mondal et al. (2014) utilizaram
ARIMA para prever preços de ações, destacando sua simplicidade e robustez na previsão de tendências.

## 2.4 SUAVIZAÇÃO EXPONENCIAL

O método de suavização exponencial tem recebido grande atenção no contexto de previsões estatísticas devido à sua eficácia, simplicidade e adaptabilidade na
previsão de séries temporais. Sua popularidade advém da capacidade intrínseca de atribuir pesos maiores às observações mais recentes em detrimento das observações
mais antigas, permitindo rápidas adaptações às mudanças na dinâmica dos dados

## (GARDNER, 1985; CIPRA, 1992).

Essa técnica tornou-se uma abordagem padrão em diversos campos práticos, incluindo gestão de estoques, controle de processos industriais, finanças e gestão de
cadeias de suprimentos. Sua ampla adoção se dá pela facilidade computacional e pela interpretação de suas previsões em comparação com métodos mais complexos
como modelos ARIMA e redes neurais (MCKENZIE, 1984).
2.4.1 Definição e estrutura do método
O método de exponential smoothing é uma técnica recursiva para previsão de séries temporais que se baseia na ponderação exponencial decrescente das
observações passadas. Formalmente, uma previsão futura é construída como uma combinação linear entre a observação mais recente e a previsão feita anteriormente.
Essa característica de atualização recursiva confere simplicidade e eficiência computacional ao método (BROWN, 1962; MCKENZIE, 1984).
Matematicamente, para o SES, a previsão do valor da série temporal 𝑋𝑡+1 pode ser expressa por:
𝑋̂𝑡+1 = 𝛼𝑋𝑡+ (1 −𝛼)𝑋̂𝑡
Onde:
•
𝑋̂𝑡+1: valor previsto para o próximo período;

---

# Page 24

24
•
𝑋𝑡: valor observado no período atual;
•
𝑋̂𝑡: previsão feita anteriormente para o período atual;
• 𝛼: constante de suavização 0 < 𝛼< 1, que define o grau de ponderação
aplicado ao dado mais recente (BROWN, 1962).
Já métodos mais avançados, como o método de Holt-Winters, consideram explicitamente os componentes de nível, tendência e sazonalidade da série temporal.
Segundo Gardner (1985), para séries com comportamento sazonal e tendência linear, a previsão futura para ℎ passos à frente é dada pela expressão geral do método de
Holt-Winters multiplicativo:
𝑋̂𝑡+ℎ= (𝐿𝑡+ ℎ × 𝑏𝑡) ×  𝑆𝑡+ℎ−𝑚(𝑘+1)
Onde:
•
𝐿𝑡 é o nível estimado da série no tempo 𝑡;
• 𝑏𝑡 é a tendência estimada no tempo 𝑡;
•
𝑆𝑡+ℎ−𝑚(𝑘+1) é o fator sazonal estimado no tempo correspondente;
• ℎ representa o horizonte futuro da previsão (quantidade de períodos à frente);
• 𝑚 é o período sazonal da série (por exemplo, 𝑚= 12 para séries mensais
anuais);
• 𝑘 é o número de ciclos completos transcorridos.
Esses métodos avançados permitem previsões mais precisas em séries complexas, com tendências claras ou padrões sazonais fortes, superando métodos
mais simples como médias móveis ou o próprio exponential smoothing simples

## (MCKENZIE, 1984; GARDNER, 1985).

2.4.2 Vantagens e limitações na previsão de dados
Entre as características fundamentais do método de exponential smoothing destacam-se:
a) Adaptabilidade: capacidade de responder rapidamente às alterações estruturais na série temporal, atribuindo pesos exponenciais aos dados
recentes (GARDNER, 1985);

---

# Page 25

25 b) Simplicidade computacional: a estrutura recursiva dos cálculos torna o
método atrativo em aplicações práticas, especialmente onde é necessária atualização constante das previsões (BROWN, 1962);
c) Flexibilidade estrutural: diferentes versões, como simples, dupla e tripla
(Holt-Winters), permitem modelar comportamentos como tendência e sazonalidade com eficiência (MCKENZIE, 1984);
d) Robustez: versões robustas do método, que usam a minimização dos desvios absolutos ou métodos M-estimadores ao invés de mínimos
quadrados, têm maior resistência a dados atípicos e séries temporais com distribuições assimétricas ou de caudas pesadas (CIPRA, 1992).
2.4.3 Aplicações e estudos de caso a) Impacto da suavização exponencial no Efeito Bullwhip: Chen, Ryan e
Simchi-Levi (2000) investigaram como a utilização do exponential smoothing na previsão de demanda pode intensificar o efeito bullwhip, fenômeno no
qual pequenas variações na demanda são ampliadas ao longo da cadeia de suprimentos. Eles demonstraram que, ao utilizar previsões com exponential
smoothing, as variações nas demandas observadas pelos fabricantes se tornam significativamente maiores do que as percebidas pelos varejistas,
aumentando os desafios de gestão e planejamento logístico nas organizações;
b) Robustez a outliers em séries temporais: Cipra (1992) avaliou o desempenho de versões robustas do método de exponential smoothing em
séries temporais contaminadas por outliers e distribuições de caudas longas.
Utilizando minimização dos desvios absolutos (norma 𝐿1) em vez dos mínimos quadrados, Cipra verificou experimentalmente que essas versões
robustas forneceram previsões significativamente mais estáveis e precisas na presença de valores extremos, superando métodos tradicionais
especialmente em séries financeiras e industriais onde valores atípicos são comuns;
c) Aplicações em controle de estoques: Gardner (1985) destacou o uso bem- sucedido de exponential smoothing no controle e previsão para gestão de
estoques. Nesse contexto, foram aplicadas variações do método para prever

---

# Page 26

26 demandas futuras e determinar níveis ótimos de estoque, reduzindo custos
relacionados à manutenção excessiva ou insuficiente de produtos em inventário. Esse exemplo demonstra claramente como o exponential
smoothing pode auxiliar gestores a otimizarem recursos financeiros e logísticos nas organizações;
d) Previsões de demanda em séries sazonais e com tendência: McKenzie
(1984) apresentou exemplos práticos demonstrando a eficácia do exponential smoothing para séries temporais com forte comportamento
sazonal e tendência definida. Em seu estudo, foi utilizado o método Holt-
Winters para capturar esses componentes, proporcionando previsões mais precisas que outros métodos tradicionais como médias móveis simples e
modelos ARIMA em séries complexas, especialmente no contexto de demanda sazonal de varejo e setores produtivos.

## 2.5 XGBOOST

O XGBoost tornou-se um dos métodos mais populares e eficazes no âmbito da previsão e classificação em machine learning, devido à sua capacidade de lidar
eficientemente com grandes quantidades de dados e produzir modelos altamente precisos. Originalmente proposto por Chen e Guestrin em 2016, o XGBoost combina
otimizações algorítmicas e técnicas avançadas de engenharia de sistemas para aprimorar significativamente o desempenho de previsões e classificações em diversas
áreas (CHEN; GUESTRIN, 2016).
2.5.1 Visão geral do Extreme Gradient Boosting
O XGBoost é uma implementação otimizada do algoritmo Gradient Boosting, baseado em árvores de decisão sequenciais. Diferentemente das abordagens
tradicionais, que utilizam árvores independentes (como o Random Forest), o XGBoost constrói árvores de maneira iterativa, com cada árvore subsequente aprendendo dos
resíduos e erros das anteriores. A combinação final das árvores resulta em um modelo robusto e altamente eficiente para prever valores futuros e classificar dados
complexos (MALIK; HARODE; KUNWAR, 2020).

---

# Page 27

27
2.5.2 Características e conceitos do XGBoost
Entre as características fundamentais do XGBoost destacam-se:
a) Boosting: Método de aprendizado de máquina que cria um modelo forte por meio da combinação sequencial de modelos fracos. Cada novo modelo tenta
corrigir os erros dos modelos anteriores (MALIK; HARODE; KUNWAR, 2020);
b) Regularização: O XGBoost incorpora penalidades ao modelo para evitar o ajuste excessivo (overfitting), limitando a complexidade através de
parâmetros como profundidade máxima das árvores, penalização por complexidade (gamma) e regularização dos pesos das folhas (lambda).
Essa abordagem resulta em modelos mais generalizáveis (CHEN;

## GUESTRIN, 2016);

c) Sparsity-aware Split Finding: Um algoritmo que otimiza o processo de divisão das árvores levando em conta a esparsidade dos dados,
economizando recursos computacionais ao ignorar valores ausentes ou zerados durante a construção das árvores (CHEN; GUESTRIN, 2016).
d) Paralelização e computação distribuída: O XGBoost é projetado para ser executado em múltiplas CPUs, permitindo o processamento paralelo dos
dados e acelerando significativamente o treinamento de grandes modelos

## (CHEN; GUESTRIN, 2016);

e) Shrinking e Column Subsampling: Técnicas adicionais que ajudam a controlar a complexidade do modelo. Shrinking reduz o impacto individual
de cada árvore, enquanto Column Subsampling seleciona aleatoriamente um subconjunto de atributos para cada árvore, aumentando a robustez e a
velocidade do modelo (CHEN; GUESTRIN, 2016).
2.5.3 Como o XGBoost prevê dados futuros
O funcionamento do XGBoost para previsões ocorre de maneira iterativa, seguindo os passos:

---

# Page 28

28 a) Inicialização: O processo se inicia com a definição de uma previsão inicial,
que geralmente corresponde à média dos valores reais presentes nos dados de treinamento, no caso de problemas de regressão. Essa previsão inicial
serve como ponto de partida para o modelo e representa a estimativa mais simples possível sem considerar ainda as relações complexas entre as
variáveis (CHEN; GUESTRIN, 2016; NIELSEN, 2016);
b) Cálculo dos resíduos: Após a obtenção da previsão inicial, calcula-se a diferença entre os valores previstos e os valores reais, gerando assim os
resíduos. Esses resíduos indicam o quanto o modelo atual está errando na previsão. O objetivo do XGBoost é reduzir esses resíduos a cada nova
iteração, corrigindo gradualmente as falhas do modelo anterior (NIELSEN, 2016; ZHANG et al., 2021);
c) Treinamento iterativo das árvores: Em cada iteração, uma nova árvore de decisão é treinada, não para prever diretamente os valores finais, mas sim
para modelar os resíduos obtidos na etapa anterior. Ou seja, cada árvore seguinte busca aprender e corrigir os erros cometidos pelo conjunto das
árvores anteriores, ajustando-se a padrões ainda não capturados (XIE;

## ZHANG, 2021; NIELSEN, 2016);

d) Atualização das previsões: As previsões do modelo são atualizadas somando as previsões das novas árvores treinadas às previsões
acumuladas das árvores anteriores. Com isso, o modelo torna-se progressivamente mais preciso a cada ciclo, pois incorpora sucessivamente
correções dos erros passados. Ao final do processo, a previsão final é composta pela soma ponderada de todas as árvores criadas durante as
iterações, representando assim uma combinação de múltiplos aprendizados parciais (CHEN; GUESTRIN, 2016; XIE; ZHANG, 2021).
A função objetivo otimizada no processo é:
𝐿(𝜑) = ∑𝑙(𝑦̂𝑦, 𝑦𝑖) 𝑖
+ ∑Ω(𝑓𝑘) 𝑘
onde:
𝑙(𝑦̂𝑦, 𝑦𝑖) representa a função de perda (e.g., erro quadrático médio);

---

# Page 29

29
Ω(𝑓𝑘) representa o termo de regularização que controla a complexidade do modelo (CHEN; GUESTRIN, 2016).
2.5.4 Exemplos práticos de uso do XGBoost a) Utilidades: Segundo Noorunnahar et al. (apud Kontopoulou et al., 2023), no
campo de utilidades, foi conduzido um estudo com o objetivo de prever a produção anual de arroz em Bangladesh. Os autores compararam a
precisão das previsões feitas por um método ARIMA otimizado, fundamentado no critério AIC, e pelo algoritmo XGBoost. Para a avaliação
dos modelos, foram consideradas métricas de erro como MAE, MPE, RMSE e MAPE. Os resultados indicaram que o modelo XGBoost obteve um
desempenho superior em relação ao ARIMA no conjunto de teste, demonstrando maior eficácia na previsão da produção de arroz para o
contexto analisado;
b) Previsão de volume de vendas no varejo: No setor de utilidades e comércio, o XGBoost tem se mostrado eficaz na previsão de volumes de vendas. A
pesquisa de Dairu e Shilong (2021) é um exemplo, onde o modelo XGBoost foi utilizado para prever o volume de vendas no varejo, comparando seus
resultados com o ARIMA clássico, o algoritmo GBDT, um modelo de LSTM e a ferramenta de previsão Prophet. Os resultados desse estudo indicaram
que as abordagens baseadas em árvores, treinadas com características de clima e temperatura, ofereceram o melhor desempenho de previsão entre os
cinco modelos, enquanto o ARIMA apresentou o pior desempenho.
Notavelmente, o XGBoost exigiu significativamente menos iterações de treinamento do que o GBDT e, juntamente com o GBDT, necessitou de
menos dados e recursos em contraste com os modelos de LSTM. Além disso, os autores propuseram um modelo de previsão de vendas baseado
em XGBoost para um conjunto de dados de bens de varejo do Walmart, demonstrando bom desempenho com menor tempo de computação e
recursos de memória.

---

# Page 30

30

## 3 METODOLOGIA

Este capítulo apresenta os procedimentos metodológicos adotados para a realização da presente pesquisa, detalhando de forma sistemática as etapas que
orientaram o desenvolvimento do estudo. São descritos o tipo de pesquisa, a abordagem utilizada, os métodos de coleta e análise dos dados, bem como os critérios
que fundamentaram as escolhas metodológicas. O objetivo é conferir transparência e fundamentação científica ao percurso investigativo, garantindo a validade e a
confiabilidade dos resultados obtidos.

## 3.1 METODOLOGIA DE TRABALHO

Com o intuito de proporcionar uma visão geral do percurso metodológico adotado, a figura a seguir apresenta, de forma esquemática, as principais etapas e
procedimentos desenvolvidos ao longo deste trabalho. O diagrama tem como objetivo ilustrar, de maneira clara e objetiva, a estrutura metodológica geral que orientou a
condução da pesquisa.
Fonte: elaborado pelo autor
Figura 1 - Metodologia geral do trabalho

---

# Page 31

31
3.1.1 Definição do problema e objetivos da previsão
Este trabalho tem como ponto de partida uma necessidade prática observada em um dos produtos desenvolvidos pela empresa onde atuo, voltado à análise e
visualização de dados corporativos. Especificamente, trata-se de um dashboard construído na ferramenta Power BI, que apresenta diversas análises de desempenho,
incluindo uma medida responsável por estimar o faturamento do mês corrente com base nos dados registrados desde o primeiro dia do mês até o momento da consulta.
O problema que este trabalho propõe a investigar consiste em avaliar se é possível aprimorar essa estimativa por meio da aplicação de modelos de aprendizado
de máquina e métodos estatísticos avançados. Para isso, foram desenvolvidos diferentes modelos preditivos (ARIMA, Theta, Suavização Exponencial e XGBoost)
utilizando os mesmos dados disponíveis no dashboard, buscando simular o contexto real de previsão. O desempenho de cada modelo foi avaliado com base em métricas
estatísticas padronizadas.
O objetivo principal deste estudo é verificar qual dos modelos testados apresenta melhor desempenho preditivo. A adoção do melhor modelo poderá resultar
em previsões mais precisas e na geração de insights mais robustos e estratégicos.
3.1.2 Coleta e pré-processamento dos dados
A coleta e o pré-processamento dos dados utilizados neste trabalho foram realizados através da ferramenta Visual Studio Code. Os dados empregados
correspondem às séries históricas de faturamento disponíveis em um produto interno da empresa, sendo originalmente utilizados em um dashboard desenvolvido em
Power BI.
Os dados utilizados neste estudo consistiram em registros transacionais de vendas contendo 37.425 transações no período de 2014 a 2025. Os campos principais
incluíram a data de emissão do pedido, valor líquido da venda, identificação do cliente e tipo de operação comercial.
O pipeline implementado seguiu uma abordagem sistemática dividida em etapas distintas, conforme mostra figura abaixo, cada uma com objetivos específicos
para preparar os dados para diferentes tipos de modelos de machine learning.

---

# Page 32

32
Fonte: elaborado pelo autor
3.1.2.1 Filtragem e agregação inicial
O processo de pré-processamento iniciou com a filtragem exclusiva de transações classificadas como "VENDA", excluindo devoluções e outros tipos de
operações comerciais. O valor líquido das vendas foi estabelecido como variável target, representando a quantidade que os modelos tentariam prever.
3.1.2.2 Anonimização dos dados
Para garantir a privacidade e conformidade com requisitos de proteção de dados, foi implementado um processo de anonimização utilizando função de hash
criptográfico MD5 para transformar identificações de clientes em códigos anônimos.
O sistema gerou identificadores no formato "CLIENTE_####" onde os quatro dígitos foram derivados deterministicamente do hash do nome original. Esta abordagem
Figura 2 - Metodologia do pré-processamento

---

# Page 33

33 protegeu a privacidade dos clientes enquanto preservou a capacidade de
rastreamento consistente ao longo do tempo.
3.1.2.3 Agregação temporal mensal
Após a filtragem inicial, os dados transacionais foram agregados temporalmente em períodos mensais, calculando a soma total de vendas para cada
mês. Este processo foi fundamental pois os modelos de séries temporais operam com observações sequenciais regularmente espaçadas no tempo.
O procedimento consistiu em agrupar todas as transações por mês e ano, gerando uma série temporal com frequência mensal cobrindo o período completo dos
dados. Cada observação representou o faturamento total do mês correspondente, resultando em aproximadamente 132 pontos temporais mensais.
3.1.2.4 Conversão para formato Darts
Os dados agregados foram então convertidos para o formato TimeSeries da biblioteca Darts, utilizada para implementação de todos os modelos neste estudo. A
biblioteca Darts oferece uma interface unificada para modelagem de séries temporais, suportando tanto métodos estatísticos tradicionais (ARIMA, Theta, Suavização
Exponencial) quanto algoritmos de machine learning (XGBoost) especializados em séries temporais.
Esta conversão incluiu a definição adequada do índice temporal (datas mensais no formato ISO), especificação da coluna de valores (faturamento mensal agregado),
e configuração da frequência da série temporal (mensal). A estrutura TimeSeries permitiu que todos os modelos acessassem funcionalidades avançadas como divisão
temporal apropriada, geração automática de features, e aplicação de transformações específicas para cada algoritmo.
3.1.2.5 Considerações sobre engenharia de features
Diferentemente de abordagens tradicionais que requerem engenharia manual extensiva de features (criação de lags, médias móveis, codificações trigonométricas

---

# Page 34

34 etc.), a biblioteca Darts realiza automaticamente a criação das features necessárias
para cada tipo de modelo durante o processo de treinamento.
Para os modelos estatísticos (ARIMA, Theta, Suavização Exponencial), a Darts opera diretamente sobre a série temporal univariada, aplicando internamente as
transformações e diferenciações necessárias.
Para o modelo XGBoost, a Darts utiliza o módulo XGBModel, que cria automaticamente features temporais através de:
a) Lags configuráveis da variável target;
b) Lags de covariadas passadas (quando aplicável);
c) Encoders temporais (mês, ano, trimestre, dia do ano, semana do ano, dia da semana);
d) Normalização apropriada via MaxAbsScaler.
Esta abordagem
simplificou significativamente
o pipeline
de pré-
processamento, eliminando a necessidade de engenharia manual de features e garantindo consistência na preparação dos dados para todos os modelos.
3.1.3 Análise exploratória e estruturação da série temporal
A análise exploratória de dados (EDA) constitui uma etapa fundamental no processo de modelagem de séries temporais, precedendo a aplicação de modelos
preditivos e fornecendo informações essenciais sobre a estrutura, padrões e características dos dados históricos. Conforme destacado por Bezerra (2006), a
compreensão adequada do comportamento temporal dos dados é crucial para a seleção e parametrização apropriada de modelos de previsão, influenciando
diretamente a qualidade e confiabilidade dos resultados obtidos.
No contexto de séries temporais de vendas, a EDA assume particular importância devido à complexidade inerente desses dados, que frequentemente
apresentam componentes de tendência, sazonalidade, ciclos econômicos e variações irregulares. Segundo Makridakis, Wheelwright e Hyndman (1999), a identificação
precisa desses componentes através de técnicas exploratórias adequadas é fundamental para orientar as decisões metodológicas subsequentes, incluindo a
escolha de modelos estatísticos apropriados e a definição de estratégias de pré- processamento.

---

# Page 35

35
3.1.3.1 Visão geral da série temporal
A análise exploratória foi implementada através de um sistema automatizado de visualizações desenvolvido em Python, utilizando bibliotecas especializadas em
análise de séries temporais. Os dados utilizados correspondem à série temporal de vendas mensais no período de outubro de 2014 a setembro de 2025, totalizando 132
observações após o pré-processamento e agregação temporal mensal.
A estruturação dos dados seguiu as diretrizes estabelecidas por Parzen (1961), que define uma série temporal como um conjunto de observações dispostas
cronologicamente, representada matematicamente como um processo estocástico.
Para garantir a adequação dos dados à análise temporal, foi implementada uma verificação rigorosa da ordenação cronológica, tratamento de valores ausentes e
validação da consistência temporal.
A primeira análise apresenta uma visão geral abrangente da série temporal, incluindo a evolução das vendas ao longo do tempo com linha de tendência,
distribuição dos valores por ano, análise das vendas acumuladas e volatilidade temporal. Esta visão panorâmica revelou uma tendência de crescimento consistente
de 2014 a 2022, seguida por um declínio significativo entre os anos 2023 e 2025, com valores variando de aproximadamente R$ 1 milhão em 2014 para um pico acima de
R$ 80 milhões em 2022.

---

# Page 36

36
Fonte: elaborado pelo autor
3.1.3.2 Decomposição STL
A decomposição STL (Seasonal-Trend using Loess) foi aplicada para separar os componentes estruturais da série temporal. A decomposição confirmou a presença
de uma tendência de longo prazo bem definida e padrões sazonais consistentes, com a série original mostrando crescimento exponencial até 2022, seguido por declínio
acentuado. O componente sazonal revelou padrões regulares de variação mensal, enquanto o resíduo indicou períodos de maior volatilidade, especialmente durante os
anos de transição econômica.
Figura 3 - Visão geral da série temporal

---

# Page 37

37
Fonte: elaborado pelo autor
3.1.3.3 Análise de sazonalidade
A análise sazonal detalhada examinou os padrões mensais e de autocorrelação da série temporal. Foram calculadas as médias mensais históricas, revelando que
determinados meses apresentam consistentemente maiores volumes de vendas. A análise de autocorrelação identificou dependências temporais significativas até o lag
12, confirmando a presença de sazonalidade anual na série.
Figura 4 – Decomposição da série temporal

---

# Page 38

38
Fonte: elaborado pelo autor
3.1.3.4 Propriedades estatísticas
A análise das propriedades estatísticas incluiu o cálculo das funções de autocorrelação (ACF) e autocorrelação parcial (PACF), fundamentais para a
parametrização de modelos ARIMA. A ACF mostrou correlações significativas nos primeiros lags, decaindo gradualmente até o lag 12, enquanto a PACF apresentou
cortes abruptos após o primeiro lag, sugerindo características autorregressivas na série. A análise da série diferenciada (primeira diferença) confirmou a remoção da
tendência, tornando a série mais adequada para modelagem estatística.
Figura 5 - Análise da sazonalidade

---

# Page 39

39
Fonte: elaborado pelo autor
3.1.3.5 Análise de distribuição
A análise de distribuição dos valores de vendas incluiu histograma com sobreposição de distribuição normal, gráfico Q-Q para teste de normalidade, box plot
para identificação de outliers, e comparação de densidade. Os resultados indicaram que a distribuição das vendas não segue uma distribuição normal, apresentando
assimetria positiva e presença de valores extremos.
Figura 6 - Propriedades estatísticas da série temporal

---

# Page 40

40
Fonte: elaborado pelo autor
3.1.3.6 Evolução temporal detalhada
A análise de evolução temporal examinou as taxas de crescimento anual, padrões sazonais por ano, e tendência linear geral. O cálculo das taxas de
crescimento revelou crescimento superior a 200% em 2015, estabilização em torno de 20 a 40% nos anos intermediários, e declínios acentuados nos anos finais.
Figura 7 - Análise de distribuição

---

# Page 41

41
Fonte: elaborado pelo autor
3.1.3.7 Análise de correlação temporal
A análise de correlação incluiu correlações com lags de 1 a 12 meses, autocorrelação parcial detalhada, matriz de correlação para lags selecionados e
correlação com componentes temporais (ano, trimestre, mês). Os resultados mostraram correlações elevadas (>0,8) para os primeiros lags, decaindo
gradualmente até o lag 12. A matriz de correlação dos lags selecionados revelou padrões de dependência temporal que orientaram a configuração dos modelos
preditivos.
Figura 8 - Evolução temporal das vendas

---

# Page 42

42
Fonte: elaborado pelo autor
3.1.3.8 Insights para modelagem
Com base nesta análise exploratória abrangente, foram identificados os seguintes resultados fundamentais para a modelagem preditiva:
a) Estacionariedade: A série original não é estacionária devido à forte tendência, requerendo diferenciação para modelos ARIMA (d = 1);
b) Sazonalidade: Presença confirmada de sazonalidade anual (período 12) com padrões consistentes;
c) Autocorrelação: Dependências temporais significativas até 12 lags, orientando a parametrização dos modelos;
d) Distribuição: Dados não seguem distribuição normal, apresentando assimetria positiva;
Figura 9 - Análise de correlação temporal

---

# Page 43

43 e) Tendência: Tendência de longo prazo bem definida com crescimento até
2022 seguido de declínio;
f) Volatilidade: Variação da volatilidade ao longo do tempo, com períodos de maior instabilidade.
Estes resultados orientaram diretamente a configuração dos parâmetros para cada modelo preditivo, a escolha das técnicas de pré-processamento específicas, e
as estratégias de validação temporal adotadas nas etapas subsequentes.

## 3.2 MODELOS DE PREVISÃO UTILIZADOS

A modelagem preditiva é a etapa central deste trabalho, sendo responsável por transformar os dados estruturados em previsões quantitativas para o faturamento.
Considerando as diferentes abordagens e características dos dados, foram selecionados múltiplos modelos de previsão, cada um com suas próprias vantagens,
desvantagens e características específicas de implementação.
Os modelos escolhidos para este estudo incluem técnicas tradicionais de séries temporais, como ARIMA, Theta e Suavização Exponencial, bem como o algoritmo
XGBoost, amplamente utilizado em aplicações empresariais para problemas de previsão com séries temporais. Cada um desses modelos foi avaliado quanto à sua
capacidade de capturar padrões históricos, prever tendências futuras e lidar com os desafios típicos desse tipo de dado, como sazonalidade, tendência e variações
irregulares.
Para garantir uma análise comparativa robusta, foram considerados fatores como a facilidade de implementação, complexidade computacional e a precisão das
previsões geradas. Todos os modelos foram implementados utilizando a biblioteca
Darts, que oferece uma interface unificada e padronizada para modelagem de séries temporais, garantindo consistência na preparação dos dados, divisão temporal e
avaliação de desempenho.
Nos subtópicos a seguir, cada modelo é apresentado individualmente, incluindo os requisitos específicos de implementação e o diagrama do fluxo metodológico
correspondente.

---

# Page 44

44

## 3.2.1 ARIMA

A figura a seguir mostra a metodologia utilizada para o modelo.
Fonte: elaborado pelo autor
3.2.1.1 Importação das bibliotecas e configuração do ambiente
A implementação do modelo ARIMA foi realizada utilizando o Visual Studio
Code como ambiente de desenvolvimento integrado, garantindo controle de versão e reprodutibilidade do código. O ambiente Python foi configurado com as seguintes
bibliotecas essenciais:
Figura 10 - Metodologia do modelo ARIMA

---

# Page 45

45 a) Darts: Biblioteca especializada em séries temporais que forneceu o módulo
ARIMA (com seleção automática de parâmetros via AutoARIMA), métodos de divisão temporal apropriados para séries temporais, e funções integradas
de avaliação e diagnóstico;
b) Pandas: Utilizado para manipulação e estruturação inicial dos dados, conversão de tipos de dados temporais, e operações de agregação e
filtragem durante o pré-processamento;
c) Matplotlib e Seaborn: Empregados para geração de visualizações diagnósticas, incluindo gráficos de série temporal, correlogramas, análise de
resíduos e comparações entre valores observados e previstos.
Esta preparação foi fundamental para garantir que todas as operações subsequentes fossem executadas de forma padronizada e rastreável.
3.2.1.2 Ingestão e conversão dos dados para série temporal
O processo de ingestão iniciou com o carregamento dos dados de faturamento mensal previamente processados na etapa 3.1.2, obtidos do arquivo CSV estruturado
com 132 observações mensais. Os dados foram validados quanto à:
a) Integridade temporal: Verificação de continuidade mensal sem lacunas, confirmação da ordenação cronológica correta, e validação do formato de
datas no padrão ISO (YYYY-MM-DD);
b) Qualidade dos valores: Identificação de valores nulos, negativos ou extremos que poderiam comprometer a modelagem, e confirmação da
escala monetária consistente (valores em reais);
c) Estrutura adequada: Configuração do índice temporal como DatetimeIndex do Pandas, garantindo operações temporais apropriadas.
A conversão para o objeto TimeSeries da Darts foi realizada especificando a coluna de valores (faturamento mensal), o índice temporal (datas mensais), e a
frequência da série ('MS' para mensal). Esta estrutura otimizada permitiu que o modelo
ARIMA acessasse funcionalidades avançadas como detecção automática de

---

# Page 46

46 periodicidade sazonal, aplicação de transformações temporais (diferenciação), e
geração de previsões de forma eficiente.
3.2.1.3 Verificação de estacionaridade e diferenciação
A avaliação de estacionariedade foi conduzida considerando os achados da análise exploratória que evidenciaram forte tendência não linear (crescimento
exponencial até 2022, seguido de declínio acentuado) e padrões sazonais anuais consistentes usando o seguinte:
a) Testes de estacionariedade: O AutoARIMA da Darts realiza testes internos
(ADF - Augmented Dickey-Fuller) para detectar a presença de raiz unitária e determinar automaticamente a necessidade de diferenciação.
b) Estratégia de diferenciação: O AutoARIMA foi configurado para explorar automaticamente:
a. Diferenciação não sazonal (d): Testadas ordens de 0 a 2, sendo d = 1
(primeira diferença) a mais comum para remover tendência linear.
b. Diferenciação sazonal (D): Avaliada com período 12 (sazonalidade anual), testando D = 0 (sem diferenciação sazonal) e D = 1 (uma
diferenciação sazonal
para remover
padrões sazonais
não estacionários).
O processo de diferenciação foi crucial para transformar a série não estacionária original em uma série com propriedades estatísticas estáveis, evitando
regressões espúrias e garantindo a validade dos pressupostos do modelo ARIMA. A biblioteca Darts aplicou estas transformações de forma automática e reversível para
as previsões finais.
3.2.1.4 Divisão dos dados em conjuntos de treino e teste
A divisão temporal foi implementada seguindo rigorosamente o princípio de não-sobreposição temporal, essencial para validação realística de modelos de séries
temporais. A estratégia adotada foi:

---

# Page 47

47 a) Conjunto de treino: Primeiros 80% da série (aproximadamente 105 meses),
representando o período de outubro de 2014 até meados de 2023. Este período incluiu a fase de crescimento consistente e o pico histórico das
vendas, fornecendo ao modelo informação suficiente sobre tendências de longo prazo e padrões sazonais estabelecidos;
b) Conjunto de teste: Últimos 20% da série (aproximadamente 27 meses), correspondendo ao período final até setembro de 2025. Este período
capturou a fase de declínio das vendas, representando um desafio real de generalização para o modelo;
c) Justificativa da divisão: A proporção 80/20 foi escolhida para garantir quantidade suficiente de dados para o treinamento (especialmente
importante para capturar múltiplos ciclos sazonais anuais), ao mesmo tempo que preservou um horizonte de teste representativo para avaliar
performance preditiva.
A implementação utilizou métodos nativos da Darts, que garantiram preservação da estrutura temporal e evitaram vazamento de informações futuras para
o conjunto de treino.
3.2.1.5 Definição dos parâmetros p, d e q
A parametrização do modelo foi conduzida através do AutoARIMA da Darts, que implementou uma busca sistemática e otimizada pelos melhores parâmetros
SARIMA(p,d,q)(P,D,Q)s. Os parâmetros foram definidos como:
b) Parâmetros não sazonais:
a. p (ordem autorregressiva): Número de lags da série defasada utilizados como preditores. Testadas ordens de 0 a 5, onde p = 1 indica
dependência do valor anterior, p = 2 inclui os dois valores anteriores etc;
b. d (ordem de diferenciação): Número de diferenciações aplicadas para tornar a série estacionária. Avaliadas ordens de 0 a 2, baseadas nos
testes de estacionariedade;

---

# Page 48

48 c. q (ordem de média móvel): Número de erros de previsão defasados
incluídos no modelo. Testadas ordens de 0 a 5, capturando dependências nos termos de erro.
c) Parâmetros sazonais (período s = 12):
a. P (autorregressivo sazonal): Dependência de valores sazonais defasados (ex.: mesmo mês do ano anterior). Testadas ordens de 0 a 2;
b. D (diferenciação sazonal): Diferenciação aplicada com período sazonal para remover não estacionariedade sazonal. Avaliadas ordens de 0 a 1;
c. Q (média móvel sazonal): Erros sazonais defasados incluídos no modelo. Testadas ordens de 0 a 2.
Para critério de seleção, o AutoARIMA utilizou o AIC (Akaike Information
Criterion) para balancear qualidade do ajuste com parcimônia do modelo, selecionando automaticamente a configuração que minimizou o AIC. O algoritmo
implementou busca
stepwise para
eficiência computacional,
explorando configurações vizinhas de forma inteligente.
3.2.1.6 Treinamento do modelo
O processo de treinamento foi executado após a seleção automática dos melhores parâmetros, utilizando os algoritmos de estimação implementados na Darts.
O treinamento envolveu:
a) Estimação por máxima verossimilhança: Os coeficientes do modelo foram estimados através da maximização da função de verossimilhança, que
encontrou os parâmetros que melhor explicaram os dados observados no conjunto de treino;
b) Otimização numérica: O processo utilizou algoritmos de otimização não linear para encontrar os valores ótimos dos coeficientes, iniciando de valores
iniciais estimados e iterando até convergência;
c) Ajuste da componente sazonal: O modelo SARIMA ajustou simultaneamente os padrões não sazonais (tendência de curto prazo, dependências de lags
próximos) e sazonais (padrões anuais, dependências de períodos equivalentes em anos anteriores);

---

# Page 49

49 d) Validação do ajuste: Durante o treinamento, foram monitoradas métricas de
convergência e estabilidade dos coeficientes estimados para garantir adequação do processo de otimização.
O resultado foi um modelo completamente parametrizado, capaz de capturar tanto as dependências temporais de curto prazo quanto os padrões sazonais anuais
identificados na análise exploratória.
3.2.1.7 Validação do modelo e ajustes finos
A etapa de validação consistiu na geração de previsões para todo o horizonte do conjunto de teste e avaliação sistemática da performance preditiva:
a) Geração de previsões: O modelo treinado foi utilizado para produzir previsões recursivas, onde cada previsão utilizou apenas informações
disponíveis até aquele ponto temporal. Este processo simulou fielmente o cenário real de previsão operacional;
b) Intervalos de confiança: Foram gerados intervalos de previsão (tipicamente
95% de confiança) baseados na variância estimada dos erros do modelo, fornecendo medida de incerteza associada a cada previsão;
c) Métricas de avaliação: A performance foi avaliada através do conjunto padronizado de métricas:
a. MAE (Mean Absolute Error): Erro absoluto médio em reais, interpretável diretamente na escala do problema;
b. RMSE (Root Mean Squared Error): Raiz do erro quadrático médio, penalizando mais fortemente grandes desvios;
c. MAPE (Mean Absolute Percentage Error): Erro percentual absoluto médio, permitindo interpretação relativa independente da escala.
d) Análise temporal das previsões: Foi conduzida análise período a período para identificar padrões nos erros, sazonalidade residual, e performance
diferencial ao longo do horizonte de previsão.

---

# Page 50

50
3.2.1.8 Análise residual
Uma análise detalhada dos resíduos do modelo foi conduzida para verificar se os erros de previsão se distribuíram de forma aleatória, sem padrões sistemáticos não
modelados. Foram gerados gráficos de autocorrelação (ACF) e autocorrelação parcial
(PACF) dos resíduos, buscando confirmar comportamento próximo ao ruído branco.
Resíduos com padrões significativos indicaram que o modelo não conseguiu capturar completamente as relações temporais nos dados. Adicionalmente, a análise
incluiu inspeção visual da distribuição dos resíduos e identificação de outliers ou eventos atípicos que poderiam comprometer a precisão das previsões futuras. Esta
validação foi essencial para confirmar a adequação do modelo selecionado.
3.2.1.9 Armazenamento dos resultados para comparação futura
Foram geradas visualizações específicas para documentar o desempenho do modelo ARIMA, incluindo gráficos de série temporal comparando valores observados
e previstos, análise de resíduos ao longo do tempo e representação gráfica da estrutura de correlação do conjunto de dados para diagnóstico.
Os resultados do modelo ARIMA, incluindo previsões, métricas de desempenho, parâmetros selecionados e diagnósticos, foram salvos de forma
estruturada para posterior comparação com os demais modelos (Theta, Suavização
Exponencial e XGBoost). Esta documentação foi essencial para a análise comparativa final e escolha da abordagem preditiva mais adequada.

---

# Page 51

51
3.2.2 Suavização Exponencial
A figura a seguir mostra a metodologia utilizada para o modelo.
Fonte: elaborado pelo autor
O modelo de Suavização Exponencial compartilhou grande parte da metodologia com o ARIMA, diferindo principalmente na abordagem de modelagem e
nos critérios de seleção do modelo. As etapas de importação de bibliotecas, ingestão e conversão de dados e a divisão treino/teste foram executadas de forma idêntica ao
ARIMA, utilizando a mesma biblioteca Darts, mesma estrutura TimeSeries, e mesma proporção 80/20 com divisão temporal rigorosa.
Figura 11 – Metodologia do modelo Suavização Exponencial

---

# Page 52

52
3.2.2.1 Análise de componentes para seleção do modelo
Diferentemente do ARIMA, que se baseou em testes de estacionariedade e análise de correlogramas, o modelo de Suavização Exponencial utilizou os resultados
da decomposição STL já realizada na análise exploratória para orientar a seleção do tipo apropriado de modelo.
Com base nos componentes já extraídos na EDA, a biblioteca Darts implementou critérios automáticos para escolha entre:
a) Suavização Exponencial Simples (SES): Para séries sem tendência ou sazonalidade significativas;
b) Método de Holt: Para séries com tendência forte, mas sazonalidade fraca;
c) Método de Holt-Winters: Para séries com ambos os componentes significativos (caso esperado desta série).
3.2.2.2 Decisão entre modelo aditivo e multiplicativo
Uma etapa específica da Suavização Exponencial foi a escolha entre formulações aditiva e multiplicativa, baseada na análise dos componentes sazonais
da EDA:
a) Modelo Aditivo: Selecionado quando a amplitude da sazonalidade permaneceu relativamente constante ao longo do tempo;
b) Modelo Multiplicativo: Selecionado quando a amplitude da sazonalidade variou proporcionalmente ao nível da série.
A decisão foi automatizada pela Darts baseada na análise da variância relativa dos componentes sazonais já extraídos na EDA.

---

# Page 53

53
3.2.2.3 Configuração e otimização de parâmetros
Ao contrário do ARIMA, que utilizou parâmetros discretos (p, d, q), a
Suavização Exponencial otimizou parâmetros contínuos de suavização:
a) Parâmetros do modelo Holt-Winters:
a. α (alfa): Parâmetro de suavização do nível (0 < α ≤ 1);
b. β (beta): Parâmetro de suavização da tendência (0 ≤ β ≤ 1);
c. γ (gama): Parâmetro de suavização sazonal (0 ≤ γ ≤ 1).
b) Período sazonal: Fixado em 12 meses conforme evidenciado na EDA;
c) Processo de otimização: A Darts utilizou algoritmos de minimização numérica para encontrar os valores ótimos que minimizaram o erro
quadrático médio no conjunto de treino.
3.2.2.4 Treinamento por suavização recursiva
O processo de treinamento diferiu fundamentalmente do ARIMA por utilizar suavização exponencial recursiva ao invés de estimação de máxima verossimilhança:
a) Inicialização dos componentes:
a. Nível inicial estimado como média dos primeiros períodos;
b. Tendência inicial calculada como diferença média inicial;
c. Índices sazonais estimados através dos primeiros ciclos da série.
b) Atualização recursiva: Para cada período t do treino, os componentes foram atualizados através de combinações ponderadas dos valores observados e
componentes anteriores projetados.
Este processo iterativo permitiu ao modelo adaptar-se gradualmente aos padrões, diferindo da estimação simultânea de todos os parâmetros no ARIMA.

---

# Page 54

54
3.2.2.5 Geração de previsões diretas
A geração de previsões na Suavização Exponencial utilizou abordagem direta
(não recursiva) baseada nos componentes finais, projetando o nível futuro adicionando tendência multiplicada pelo horizonte ao último nível, e obtendo o
componente sazonal do índice correspondente ao período do ano.
3.2.2.6 Análise residual específica para suavização
A análise residual seguiu protocolo similar ao ARIMA, mas com focos específicos na validação de componentes (tendência e sazonalidade), estabilidade
dos parâmetros otimizados (α, β e γ), e adequação do modelo selecionado (aditivo vs.
multiplicativo) através de análise visual dos resíduos padronizados e métricas de ajuste.
3.2.3 Theta
O modelo Theta compartilhou as etapas fundamentais de preparação com os modelos anteriores, diferindo principalmente na abordagem de decomposição e
extrapolação. As etapas de importação de bibliotecas, ingestão e conversão de dados e divisão treino/teste foram executadas de forma idêntica aos modelos anteriores,
utilizando a mesma biblioteca Darts, mesma estrutura TimeSeries, e mesma divisão temporal 80/20.
A figura a seguir mostra a metodologia utilizada para o modelo.

---

# Page 55

55
Fonte: elaborado pelo autor
3.2.3.1 Verificação de pré-condições do método Theta
O método Theta na biblioteca Darts exigiu verificações específicas antes da aplicação:
a) Validação da série temporal: Confirmação da ausência de valores nulos na série, pois o Theta da Darts não possui tratamento automático para dados
ausentes;
Figura 12 – Metodologia do modelo Theta

---

# Page 56

56 b) Verificação de univariância: O método foi aplicado exclusivamente à série
temporal univariada de faturamento mensal, sem variáveis explicativas adicionais, seguindo a natureza original do método proposto por
Assimakopoulos e Nikolopoulos (2000);
c) Confirmação de regularidade temporal: Verificação da frequência mensal constante da série, requisito para a decomposição Theta funcionar
adequadamente.
3.2.3.2 Configuração automática vs. manual do modelo
No quesito de configuração, o método Theta da Darts ofereceu configuração totalmente automática:
a) Parâmetro Theta (θ): A Darts implementou seleção automática do parâmetro θ, que controla a curvatura das linhas Theta. Valores θ < 1 enfatizam
tendências de longo prazo, enquanto θ > 1 destacam variações de curto prazo;
b) Detecção automática de sazonalidade: O Theta detectou automaticamente a presença e o período da sazonalidade (12 meses) com base nos padrões
da série;
c) Configuração de decomposição: O modelo foi configurado para aplicar decomposição automática da série em componentes Theta, sem
necessidade de especificação manual.
3.2.3.3 Decomposição e criação das linhas Theta
Esta etapa foi específica do método Theta, onde os seguintes pontos foram realizados:
a) Aplicação das segundas diferenças: O método aplicou o operador de segundas diferenças à série original conforme a formulação matemática de
Assimakopoulos e Nikolopoulos (2000);
b) Geração das linhas Theta: Foram criadas múltiplas linhas Theta através de transformações matemáticas, incluindo:

---

# Page 57

57 a. Linha Theta 0 (θ = 0): Representa tendência linear de longo prazo
b. Linha Theta 2 (θ = 2): Captura variações de curto prazo e sazonalidade.
3.2.3.4 Treinamento e ajuste das componentes
O processo de treinamento do Theta diferiu dos outros modelos no seguinte:
a) Ajuste das linhas individuais: Cada linha Theta foi ajustada separadamente:
a. Linha Theta 0: Ajustada por regressão linear para capturar tendência de longo prazo;
b. Linha Theta 2: Ajustada por Suavização Exponencial Simples (SES) para variações de curto prazo.
b) Otimização automática: A Darts implementou otimização automática dos parâmetros de cada componente.
3.2.3.5 Combinação de previsões e extrapolação
A geração de previsões seguiu abordagem única de combinação de extrapolações, onde cada linha Theta foi extrapolada separadamente para o horizonte
de teste, e as previsões finais foram obtidas através de combinação ponderada das extrapolações individuais, tipicamente com pesos iguais ou otimizados baseados na
performance histórica.
3.2.3.6 Avaliação e diagnósticos específicos
A avaliação seguiu protocolo similar aos modelos anteriores, com análises específicas de validação das linhas Theta, verificação da capacidade de reconstrução
da série original, e análise de estabilidade dos parâmetros otimizados.

---

# Page 58

58
3.2.4 XGBoost
A figura 3 mostra a metodologia utilizada para o modelo.
Fonte: elaborado pelo autor
3.2.4.1 Preparação e integração com Darts
O modelo XGBoost foi implementado utilizando o módulo `XGBModel` da biblioteca Darts, que integra o algoritmo XGBoost com a infraestrutura de séries
temporais da Darts. Diferentemente da implementação tradicional que requer engenharia manual extensiva de features, o `XGBModel` da Darts automatiza a
criação de features temporais necessárias para o treinamento.
A entrada do modelo foi a mesma série temporal univariada utilizada pelos outros modelos (faturamento mensal agregado), mantendo consistência na
preparação dos dados. A Darts se encarregou automaticamente de transformar esta
Figura 13 – Metodologia do modelo XGBoost

---

# Page 59

59 série temporal em formato tabular apropriado para o XGBoost durante o processo de
treinamento.
3.2.4.2 Divisão dos dados em treino e teste Engenharia automática de features
Assim como nos demais modelos, os dados foram divididos respeitando rigorosamente a ordem cronológica na proporção 80/20, evitando vazamento de
informações futuras. A Darts garantiu que a divisão temporal fosse consistente com os outros modelos implementados.
3.2.4.3 Engenharia automática de features
O XGBModel da Darts criou automaticamente as features necessárias através de parâmetros configuráveis:
a) Lags da variável target: Foram configurados 17 lags principais [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -15, -18, -24, -30, -36] para capturar
dependências temporais em diferentes horizontes.
b) Lags de covariadas passadas: Configurados 8 lags [-1, -2, -3, -4, -5, -6, -12, -24] para capturar padrões adicionais de dependência temporal.
c) Encoders temporais: Foram adicionados automaticamente 6 encoders temporais (mês, ano, trimestre, dia do ano, semana do ano, dia da semana)
para capturar padrões cíclicos e sazonais.
d) Normalização: Aplicada automaticamente via MaxAbsScaler para garantir escala apropriada das features, particularmente importante para lidar com
outliers em dados de vendas.
Esta abordagem eliminou a necessidade de criar manualmente features como médias móveis, codificações trigonométricas, e estatísticas agregadas, simplificando
significativamente o pipeline e garantindo que apenas as features mais relevantes fossem utilizadas.

---

# Page 60

60
3.2.4.4 Configuração dos hiper parâmetros iniciais
O modelo XGBoost implementado via Darts separou os parâmetros em duas categorias distintas: parâmetros específicos do framework Darts para processamento
de séries temporais e hiper parâmetros do algoritmo XGBoost propriamente dito.
a) Parâmetros do framework Darts (configuração de séries temporais):
a. lags: 17 valores de defasagem [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -
12, -15, -18, -24, -30, -36] para capturar dependências temporais em múltiplos horizontes.
b. lags_past_covariates: 8 lags adicionais [-1, -2, -3, -4, -5, -6, -12, -24] para padrões de dependência temporal complementares.
c. add_encoders: Encoders temporais automáticos incluindo mês, ano, trimestre, dia do ano, semana do ano e dia da semana para captura de
padrões cíclicos e sazonais.
d. data_scaling:
MaxAbsScaler aplicado
automaticamente para
normalização robusta das features.
b) Hiper parâmetros do algoritmo XGBoost (passados via kwargs):
a. n_estimators: 2000 arvores de decisão para garantir capacidade adequada de aprendizado e convergência do algoritmo de gradient
boosting.
b. max_depth: 8 níveis de profundidade máxima, controlando a complexidade das arvores individuais e evitando overfitting.
c. learning_rate: 0.05 para controlar o peso de cada nova arvore no ensemble, garantindo aprendizado estável e convergência gradual.
d. subsample: 0.9 (90% de amostragem) para aumentar a generalização do modelo através de variação estocástica nas amostras de
treinamento.
e. colsample_bytree: 0.9 para selecionar aleatoriamente 90% das features em cada arvore, promovendo diversidade no ensemble.
f. reg_alpha: 0.2 (regularização L1/Lasso) para penalizar complexidade e promover esparsidade nos pesos do modelo.

---

# Page 61

61 g. reg_lambda: 1.5 (regularização L2/Ridge) para controle adicional de
complexidade e suavização dos pesos.
h. random_state: 42 para garantir reprodutibilidade total dos resultados entre execuções.
Esta configuração híbrida aproveitou a especialização da Darts em processamento de séries temporais (geração automática de lags e encoders
temporais) combinada com o poder preditivo do algoritmo XGBoost (ensemble de arvores com gradient boosting). Os hiper parâmetros do XGBoost foram definidos
manualmente com base em práticas estabelecidas para modelos de previsão, priorizando capacidade de aprendizado (n_estimators alto e max_depth moderado)
equilibrada com regularização (reg_alpha e reg_lambda) para evitar overfitting.
3.2.4.5 Treinamento do modelo
O processo de treinamento do XGBoost seguiu o paradigma de gradient boosting:
a) Inicialização: O processo iniciou com uma previsão inicial simples
(geralmente a média dos valores de treino);
b) Treinamento iterativo: Em cada iteração, uma nova árvore de decisão foi treinada para modelar os resíduos (erros) das árvores anteriores, corrigindo
gradualmente as falhas do modelo;
c) Atualização das previsões: As previsões foram atualizadas somando as previsões das novas árvores às previsões acumuladas das árvores
anteriores, multiplicadas pela taxa de aprendizado (learning_rate);
d) Regularização: Durante o treinamento, os termos de regularização L1 e L2 foram aplicados para penalizar complexidade excessiva e promover
modelos mais simples e generalizáveis.
A integração com Darts garantiu que todo este processo respeitasse a natureza temporal dos dados, utilizando apenas informações disponíveis até cada ponto
temporal durante o treinamento.

---

# Page 62

62
3.2.4.6 Avaliação inicial de desempenho
A avaliação do desempenho foi realizada de maneira análoga aos outros modelos, através das métricas MAE, RMSE e MAPE aplicadas ao conjunto de teste.
A análise dos erros permitiu verificar a capacidade do modelo em capturar padrões complexos presentes nos dados de vendas.
3.2.4.7 Validação e análise de resultados
Foi empregada validação temporal adequada a séries temporais, assegurando a robustez dos resultados e a ausência de overfitting. Os resultados da validação
foram analisados quanto à consistência e possíveis padrões residuais, confirmando a adequação do modelo.
3.2.4.8 Geração das previsões finais e armazenamento dos resultados
As previsões finais geradas pelo modelo XGBoost foram armazenadas em formato estruturado para comparação direta com os resultados dos demais modelos
(ARIMA, Theta e Suavização Exponencial), permitindo análise comparativa abrangente baseada nas mesmas métricas padronizadas.

## 3.2.5 Metodologia do método híbrido no Power BI

O método de previsão atualmente implementado no Power BI utiliza uma abordagem híbrida que combina dois métodos estatísticos simples, mas robustos, para gerar
previsões de faturamento mensal. Este método foi considerado como baseline
(solução de referência) para comparação com os modelos de machine learning desenvolvidos neste trabalho.

### 3.2.5.1 Estrutura da solução no Power BI

A solução foi implementada através de medidas DAX no Power BI, que realizam cálculos automáticos a partir dos dados de faturamento armazenados no banco de
dados corporativo. O processo segue a seguinte estrutura:
a) **Medida de Faturamento Realizado**: Calcula a soma do faturamento líquido mensal, considerando apenas operações de vendas processadas (OPERACAO = "VENDA")
que geraram cobrança (GERA_COBRANCA = 1). Esta medida filtra automaticamente os dados pela dimensão de data selecionada.
b) **Medida de Faturamento Mensal**: Agrega o faturamento realizado para cada mês/ano, garantindo que cada ponto temporal seja associado ao valor correspondente
de faturamento.
c) **Período de Teste**: Define o intervalo temporal para as previsões (julho de
2023 a setembro de 2025), correspondendo aos 27 meses utilizados para validação dos modelos.

### 3.2.5.2 Cálculo da previsão híbrida

O método híbrido combina duas técnicas estatísticas com igual peso (50% cada):
a) **Média Movel de 6 Meses (MM6)**: Calcula a média aritmética dos 6 meses anteriores ao período de previsão. Esta técnica captura tendências recentes e
variabilidades de curto prazo nos dados de vendas, reduzindo o impacto de flutuações aleatórias.
b) **Year-over-Year (YoY)**: Utiliza o valor de faturamento do mesmo mês no ano anterior. Esta técnica captura padrões sazonais anuais, presumindo que os padrões
de vendas se repetem em ciclos anuais.
O cálculo final da previsão híbrida para cada mês e dado por:
Previsão Híbrida = (Média Movel 6 Meses × 0.5) + (Year-over-Year × 0.5)
Esta combinação permite que o modelo capture tanto tendências recentes (via MM6) quanto padrões sazonais (via YoY), equilibrando a adaptabilidade a mudanças
curtas com a estabilidade de padrões históricos anuais.

### 3.2.5.3 Extração de dados e comparação com modelos de ML

Após a implementação das medidas DAX no Power BI, foi criada uma tabela contendo as seguintes colunas para cada mês no período de teste:
- **Mês/Ano**: Identificador temporal do período;
- **Faturamento Real Teste**: Valor observado de faturamento para o mês, obtido através da medida de Faturamento Realizado;
- **Faturamento Previsto (Híbrido)**: Valor previsto usando o método híbrido descrito acima.
Esta tabela foi extraída do Power BI em formato CSV e importada em um script
Python para realizar a comparação com o melhor modelo de machine learning
(XGBoost). As métricas padrão (MAE, RMSE e MAPE) foram calculadas diretamente sobre essa tabela, permitindo uma avaliação quantitativa equivalente à realizada
para os demais modelos de previsão desenvolvidos neste trabalho.
O armazenamento dos dados em formato CSV garante rastreabilidade dos resultados e facilita a reprodutibilidade da análise, permitindo futuras revisões e
melhorias na metodologia de comparação.

## 3.3 AVALIAÇÃO E COMPARAÇÃO DOS MODELOS

Após o ajuste e validação de todos os modelos preditivos desenvolvidos neste trabalho, foi realizada uma comparação quantitativa do desempenho de cada
modelo utilizando as seguintes métricas estatísticas, recomendadas pela literatura para problemas de previsão de séries temporais:
a) Erro Médio Absoluto (MAE);
b) Raiz do Erro Quadrático Médio (RMSE);
c) Erro Percentual Absoluto Médio (MAPE).

### 3.3.1 Comparação entre os modelos de aprendizado de máquina e estatísticos

As métricas foram calculadas para o conjunto de teste de cada modelo (27 meses, de julho de 2023 a setembro de 2025), permitindo uma avaliação consistente e
comparável. Este processo seguiu a metodologia recomendada pela literatura para comparação de modelos de previsão de séries temporais, conforme Hyndman et al.
(1999) e Gardner (1985).
Os quatro modelos implementados (ARIMA, Exponential Smoothing, Theta e XGBoost) foram submetidos ao mesmo conjunto de teste, com as mesmas métricas de erro
calculadas de forma padronizada. Esse procedimento permitiu uma comparação quantitativa objetiva do desempenho de cada abordagem, considerando tanto modelos
estatísticos tradicionais quanto algoritmos de aprendizado de máquina.

### 3.3.2 Comparação do melhor modelo de ML versus método Power BI

Na sequência, o modelo de melhor desempenho entre os algoritmos de machine learning foi comparado diretamente ao método de previsão híbrido atualmente
implementado no Power BI. Esta comparação adicional é essencial para responder à questão central de pesquisa deste trabalho: se modelos avançados de aprendizado de
máquina conseguem superar uma abordagem estatística simples mas consolidada.
O procedimento de comparação seguiu a mesma metodologia utilizada para a comparação entre os modelos de ML. As métricas MAE, RMSE e MAPE foram
calculadas para ambos os modelos utilizando o mesmo período de teste (27 meses, de julho de 2023 a setembro de 2025), permitindo uma avaliação consistente e
direta da efetividade de cada abordagem.

## 4 RESULTADOS

Nesta seção são apresentados os resultados da implementação e avaliação de todos os modelos de previsão desenvolvidos neste trabalho. Os resultados são organizados
por modelo, apresentando as métricas de desempenho obtidas no conjunto de teste
(27 meses, de julho de 2023 a setembro de 2025) para cada abordagem.

### 4.1 Resultados dos Modelos de Aprendizado de Máquina e Estatísticos

Esta subseção detalha o desempenho de cada técnica (ARIMA, Exponential Smoothing, Theta e XGBoost), organizados em ordem de sofisticação crescente. A apresentação segue uma progressão que permite visualizar claramente como a capacidade preditiva evoluiu entre os modelos tradicionais estatísticos até as abordagens mais avançadas de machine learning.

#### 4.1.1 ARIMA

O modelo ARIMA implementado utilizando AutoARIMA da biblioteca Darts apresentou os seguintes resultados no conjunto de teste:

| Métrica | Valor |
|---------|-------|
| MAE | R$ 28,710,800.45 |
| RMSE | R$ 32,311,238.76 |
| MAPE | 78.73% |

O ARIMA apresentou o desempenho mais inferior entre todos os modelos testados.
O modelo dificuldade em capturar adequadamente os padrões complexos presentes nos dados de vendas, resultando em erros absolutamente elevados. O MAPE de 78.73%
indica que, em média, as previsões do ARIMA desviaram 78.73% dos valores reais observados, demonstrando capacidade preditiva muito limitada para este conjunto
de dados específico.

#### 4.1.2 Exponential Smoothing

O modelo de Suavização Exponencial (Exponential Smoothing) foi implementado utilizando a classe ExponentialSmoothing da Darts, aplicando o método de
Holt-Winters para capturar componentes de nível, tendência e sazonalidade.
Os resultados obtidos foram:

| Métrica | Valor |
|---------|-------|
| MAE | R$ 21,846,386.39 |
| RMSE | R$ 25,914,518.67 |
| MAPE | 63.00% |

O Exponential Smoothing apresentou desempenho superior ao ARIMA em todas as métricas, reduzindo o erro em 23.8% em relação ao MAE do ARIMA. Apesar dessa
melhoria, o MAPE de 63.00% ainda indica erros significativos nas previsões. O modelo demonstrou melhor capacidade que o ARIMA em capturar a sazonalidade dos
dados, porém ainda se mostrou insuficiente para gerar previsões com acurácia adequada para fins organizacionais.

#### 4.1.3 Theta

O método Theta foi implementado utilizando a classe AutoTheta da Darts, que aplica automaticamente técnicas de decomposição temporal para capturar padrões de longo
e curto prazo na série. Os resultados foram:

| Métrica | Valor |
|---------|-------|
| MAE | R$ 17,327,600.78 |
| RMSE | R$ 21,287,394.49 |
| MAPE | 39.71% |

O Theta apresentou desempenho substancialmente melhor que os dois modelos anteriores, reduzindo o MAE em 39.6% em relação ao Exponential Smoothing e em
69.2% em relação ao ARIMA. O MAPE de 39.71% representa uma redução significativa na magnitude dos erros percentuais, indicando que o método Theta conseguiu
capturar melhor os padrões sazonais e de tendência presentes nos dados de faturamento.

#### 4.1.4 XGBoost

O modelo XGBoost foi implementado utilizando a classe XGBModel da Darts com configuração de 17 lags principais, 8 lags de covariadas passadas e 6 encoders
temporais, combinados com hyperparâmetros otimizados para o problema específico.
Os resultados obtidos foram:

| Métrica | Valor |
|---------|-------|
| MAE | R$ 10,110,160.96 |
| RMSE | R$ 13,302,309.10 |
| MAPE | 26.91% |

O XGBoost apresentou o melhor desempenho entre todos os modelos de machine learning testados. O modelo reduziu o MAE em 41.7% em relação ao Theta, 53.7%
em relação ao Exponential Smoothing e 64.8% em relação ao ARIMA. O MAPE de
26.91% representa a menor taxa de erro percentual observada entre os modelos, indicando que o XGBoost conseguiu capturar de forma mais eficaz os padrões
complexos e não-lineares presentes nos dados de vendas. A superioridade do
XGBoost reflete a capacidade do algoritmo de gradient boosting em modelar relacionamentos complexos através de suas múltiplas features de engenharia
temporal.

### 4.2 Resumo Comparativo dos Modelos de ML

A tabela abaixo apresenta um resumo consolidado dos resultados de todos os modelos de machine learning e estatísticos:

| Modelo | MAE (R$) | RMSE (R$) | MAPE (%) | Ranking |
|--------|----------|-----------|----------|---------|
| ARIMA | 28,710,800.45 | 32,311,238.76 | 78.73 | 4º |
| Exponential Smoothing | 21,846,386.39 | 25,914,518.67 | 63.00 | 3º |
| Theta | 17,327,600.78 | 21,287,394.49 | 39.71 | 2º |
| XGBoost | 10,110,160.96 | 13,302,309.10 | 26.91 | 1º |

Com base nesta análise, o XGBoost foi selecionado como o melhor modelo entre os algoritmos de machine learning testados para a próxima etapa de comparação com a
abordagem implementada no Power BI.

### 4.3 Comparação: Melhor Modelo de ML (XGBoost) versus Método Power BI

Após identificar o XGBoost como o melhor modelo entre os algoritmos de machine learning testados, foi realizada uma comparação direta com o método de previsão
híbrido implementado no Power BI. Esta comparação é fundamental para responder à questão de pesquisa central deste trabalho.
Os resultados obtidos foram os seguintes:

| Método/Modelo | MAE (R$) | RMSE (R$) | MAPE (%) | Viés (%) | Acurácia (%) |
|---------------|----------|-----------|----------|----------|--------------|
| Power BI (Híbrido: MM6 + YoY) | 95,139.69 | 119,785.85 | 23.45 | 1.47 | 79.44 |
| XGBoost (Melhor ML) | 10,110,160.96 | 13,302,309.10 | 26.91 | - | - |

**Resultado Principal: O método Power BI apresentou desempenho superior ao modelo XGBoost em todas as métricas de erro avaliadas.**
O método híbrido implementado no Power BI combina duas técnicas estatísticas simples com efetividade comprovada:
1. **Média Móvel 6 Meses (MM6)**: Calcula a média aritmética dos últimos 6 meses, capturando tendências recentes e flutuações de curto prazo;
2. **Year-over-Year (YoY)**: Utiliza o valor de faturamento do mesmo mês no ano anterior, assumindo que os padrões de vendas se repetem anualmente (sazonalidade anual bem definida).
Cada componente contribui com peso igual (50% cada), resultando em previsões que balanceiam adaptabilidade a mudanças recentes com a estabilidade de padrões históricos estabelecidos.
O desempenho superior é evidente em todas as métricas:
- **MAPE**: Power BI obteve 23.45% enquanto XGBoost alcançou 26.91%, representando uma diferença de 3.46 pontos percentuais, equivalente a uma melhoria relativa de
12.9% em favor do Power BI;
- **MAE**: Power BI obteve R$ 95,139.69 enquanto XGBoost alcançou R$ 10,110,160.96, uma diferença absoluta de R$ 10,015,021.27, equivalente a uma melhoria relativa de
99.1% em favor do Power BI;
- **RMSE**: Power BI obteve R$ 119,785.85 enquanto XGBoost alcançou
R$ 13,302,309.10, indicando que o Power BI produz erros muito mais concentrados e previsíveis.
Adicionalmente, o método Power BI apresentou viés de apenas 1.47%, indicando que as previsões são praticamente não-enviesadas em média, e acurácia de 79.44%
quando considerado o MAE em relação à média dos valores observados no período.

## 5 DISCUSSÃO

Os resultados apresentados na seção anterior revelam insights importantes sobre a efetividade relativa de diferentes abordagens de previsão quando aplicadas ao
contexto específico de previsão de vendas nesta organização.

### 5.1 Desempenho Relativo dos Modelos de Machine Learning

A hierarquia de desempenho observada entre os modelos de machine learning e estatísticos (ARIMA < Exponential Smoothing < Theta < XGBoost) reflete
progressivamente maior sofisticação na captura de padrões temporais. O ARIMA, apesar de sua importância teórica como modelo canônico em análise de séries
temporais, apresentou desempenho inadequado para este conjunto de dados, possivelmente devido a sua limitação em modelar relações não-lineares complexas.
O Exponential Smoothing demonstrou melhor capacidade em capturar a sazonalidade através do método de Holt-Winters, porém ainda apresentou limitações. O Theta,
com sua abordagem de decomposição temporal, melhorou substancialmente o desempenho, sugerindo que a separação explícita de componentes de longo e curto
prazo é mais eficaz para estes dados que modelos globais como ARIMA.
O XGBoost, finalmente, apresentou o melhor desempenho entre os modelos testados.
A superioridade do XGBoost pode ser atribuída à sua capacidade de:
- Modelar interações não-lineares entre features através de múltiplas árvores de decisão;
- Capturar padrões complexos através de 17 lags principais, 8 lags de covariadas e 6 encoders temporais;
- Aplicar regularização L1 e L2 para evitar overfitting mantendo capacidade preditiva.
Contudo, o fato de que um modelo com 17 lags e 6 encoders temporais foi superado por uma combinação simples de dois métodos estatísticos básicos é revelador.

### 5.2 Superioridade do Método Power BI: Uma Análise Crítica

O resultado mais significativo deste estudo é que o método híbrido do Power BI superou substancialmente o melhor modelo de machine learning testado. Esta
descoberta contraria a expectativa comum de que algoritmos mais complexos e sofisticados produzem necessariamente melhores previsões.
O método Power BI combina apenas dois componentes simples, mas poderosos:
- **Média Móvel 6 Meses**: Captura tendências recentes e volatilidades de curto prazo, sendo responsiva a mudanças de comportamento nos últimos meses. Por exemplo, se houve uma queda de vendas nos últimos 3 meses, a MM6 refletirá essa redução;
- **Year-over-Year (YoY)**: Baseia-se na premissa de que padrões de vendas se repetem em ciclos anuais. Concretamente, para prever faturamento em junho de 2024, utiliza-se o valor de junho de 2023, capturando sazonalidade estabelecida. Esta abordagem é particularmente eficaz quando o negócio tem comportamentos sazonais bem definidos (p. ex., picos de vendas em períodos específicos do ano que se repetem consistentemente).
A melhoria de 12.9% do Power BI sobre o XGBoost em MAPE, embora menor em termos percentuais, é mais significativa quando considerado em contexto:
1. **Simplicidade Operacional**: O método Power BI é imediatamente compreensível pelos stakeholders não-técnicos, facilitando aceitação organizacional e auditoria
das previsões;
2. **Estabilidade**: A ausência de hiperparâmetros complexos reduz o risco de degradação de desempenho em futuras aplicações ou quando novos dados são
incorporados;
3. **Interpretabilidade**: Cada componente (MM6 e YoY) é facilmente explicável em reuniões de negócio, ao contrário do XGBoost que funciona como "caixa preta";
4. **Manutenção**: O método Power BI requer pouca ou nenhuma reentrenagem, enquanto modelos de ML requerem monitoramento contínuo de performance e
possível reciclagem periódica.
Estes fatores sugerem que a superioridade observada do Power BI não é apenas uma questão estatística, mas também de adequação prática às necessidades
organizacionais.

### 5.3 Implicações para a Prática Organizacional

Este estudo demonstra que para o contexto específico de previsão de vendas desta organização, métodos simples e bem estabelecidos podem superar algoritmos
sofisticados de machine learning. Este achado alinha-se com a literatura que sugere que métodos estatísticos tradicionais frequentemente alcançam desempenho
competitivo em cenários com sazonalidade clara e padrões bem definidos, conforme observado em estudos como Makridakis et al. (2000) e mais recentemente em
Hyndman e Athanasopoulos (2021).
As implicações práticas incluem:
1. A continuação do método Power BI atual é justificada tanto estatística quanto operacionalmente;
2. Investimentos em infraestrutura de ML complexo para este problema específico não se justificam baseado em ganhos de acurácia;
3. Recursos podem ser melhor alocados em outras áreas onde ML pode oferecer vantagens mais substantivas.
Contudo, reconhece-se que diferentes segmentos de produtos ou períodos temporais específicos podem apresentar padrões distintos, sugerindo que análises
segmentadas poderiam ser exploradas em trabalhos futuros.

## 6 CONCLUSÃO

Este trabalho apresentou uma análise comparativa abrangente de diferentes abordagens para previsão de vendas, comparando modelos estatísticos tradicionais
(ARIMA, Exponential Smoothing, Theta) com algoritmos modernos de machine learning (XGBoost) e o método híbrido atualmente implementado no Power BI.

### 6.1 Síntese dos Achados Principais

Os principais achados deste estudo foram:
1. **Hierarquia de Desempenho ML**: Entre os modelos de machine learning e estatísticos testados, o XGBoost apresentou o melhor desempenho com MAPE de
26.91%, seguido por Theta (39.71%), Exponential Smoothing (63.00%) e ARIMA
(78.73%);
2. **Superioridade do Power BI**: O método híbrido Power BI (MM6 + YoY) superou o melhor modelo de ML em todas as métricas, alcançando MAPE de 23.45% comparado
aos 26.91% do XGBoost, uma melhoria relativa de 12.9%;
3. **Simplicidade Vence Complexidade**: A combinação simples de dois métodos estatísticos básicos provou ser mais eficaz que um algoritmo sofisticado com 17
lags, 8 covariadas e 6 encoders temporais, sugerindo adequação particularmente alta do Power BI aos padrões específicos dos dados de faturamento;
4. **Viabilidade Operacional**: Além do desempenho superior, o método Power BI oferece vantagens significativas em termos de interpretabilidade, estabilidade,
manutenção e aceitação organizacional.

### 6.2 Resposta à Questão de Pesquisa

A questão central deste trabalho foi: "Modelos avançados de aprendizado de máquina conseguem superar o método híbrido simples implementado no Power BI?"
**Resposta: Não.** Os modelos testados, inclusive o melhor algoritmo de ML (XGBoost), não conseguiram superar a efetividade do método Power BI. Ao contrário, o Power BI
apresentou desempenho superior, consolidando sua adequação como método de previsão para este contexto específico.

### 6.3 Recomendações

Com base nos achados deste estudo, recomenda-se:
1. **Manutenção do Método Atual**: Continuar utilizando o método híbrido Power BI para previsões de faturamento mensal, dado seu desempenho superior e
características operacionais favoráveis;
2. **Não Investir em ML Geral**: Desconsiderar investimentos em implementação de modelos de ML complexos para o problema geral de previsão de vendas, pois os
ganhos incrementais de acurácia não justificam a complexidade adicional;
3. **Análise Segmentada**: Explorar em trabalhos futuros se diferentes categorias de produtos ou períodos sazonais específicos apresentam padrões que possam
beneficiar de abordagens diferenciadas;
4. **Monitoramento Contínuo**: Manter monitoramento das métricas de desempenho do método Power BI, com reavaliação periódica caso novos dados ou mudanças
estruturais nos padrões de vendas sejam observados.

### 6.4 Contribuição Científica

Este trabalho contribui para a literatura de previsão de séries temporais ao:
- Documentar empiricamente um cenário onde métodos simples superam algoritmos sofisticados;
- Destacar a importância de considerar não apenas acurácia estatística, mas também viabilidade operacional na seleção de métodos de previsão;
- Reforçar a recomendação de Makridakis et al. de não negligenciar métodos estatísticos simples em favor de abordagens mais complexas sem evidências claras
de ganho significativo;
- Contribuir para a aproximação entre pesquisa acadêmica e prática organizacional ao demonstrar que a "melhor" solução frequentemente não é a mais sofisticada
tecnicamente.

### 6.5 Limitações e Trabalhos Futuros

Este estudo apresenta algumas limitações:
1. **Período de Dados Limitado**: Os dados cobrem apenas 132 observações mensais
(2014-2025), o que pode não ser suficiente para capturar padrões em ciclos econômicos muito longos;
2. **Ausência de Variáveis Exógenas**: O estudo utilizou apenas a série temporal de faturamento sem incorporar variáveis exógenas como sazonalidade de
campanhas, indicadores econômicos ou fatores externos;
3. **Contexto Organizacional Específico**: Os achados são específicos a este conjunto de dados e organização, podendo não ser generalizáveis para outros
contextos.
Trabalhos futuros poderiam explorar:
- Incorporação de variáveis exógenas nos modelos de ML;
- Análise segmentada por categoria de produto ou unidade de negócio;
- Exploração de abordagens de ensemble que combinem Power BI com modelos ML;
- Investigação de métodos ML mais avançados (redes neurais, LSTM) para comparação;
- Análise de horizonte de previsão: avaliar se XGBoost se desempenha melhor em previsões de longo prazo.

### 6.6 Observações Finais

Este trabalho reafirma uma lição importante para profissionais de dados e cientistas de dados: a escolha do melhor método nem sempre favorece a solução mais sofisticada.
Efetividade prática, interpretabilidade, estabilidade e adequação ao contexto organizacional são critérios igualmente importantes que devem ser considerados
quando se seleciona abordagens para problemas reais de previsão.

## REFERÊNCIAS

ASSIMAKOPOULOS, V.; NIKOLOPOULOS, K. The Theta model: a decomposition approach to forecasting. International Journal of Forecasting, v. 16, n. 4, p. 521–
530, out. 2000. Disponível em: https://doi.org/10.1016/S0169-2070(00)00066-2.

---

# Page 64

64
BEZERRA, Manoel Ivanildo Silvestre. Apostila de Análise de Séries Temporais.
São Paulo: UNESP, 2006. Disponível em:
https://www.ibilce.unesp.br/Home/Departamentos/MatematicaEstatistica/apostila_ser ies_temporais_unesp.pdf.
BOX, G. E. P. et al. Time séries analysis: forecasting and control. Hoboken, New
Jersey: John Wiley & Sons, 2015.
CHEN, T.; GUESTRIN, C. XGBoost: a Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery
and Data Mining - KDD ’16, v. 1, n. 1, p. 785–794, 13 ago. 2016. Disponível em:
https://doi.org/10.1145/2939672.2939785.
DAIRU, X.; SHILONG, Z. Machine Learning Model for Sales Forecasting by
Using XGBoost. Disponível em:
https://doi.org/10.1109/ICCECE51280.2021.9342304.
ENSAFI, Y. et al. Time-séries forecasting of seasonal items sales using machine learning – A comparative analysis. International Journal of Information
Management Data Insights, v. 2, n. 1, p. 100058, abr. 2022. Disponível em:
https://doi.org/10.1016/j.jjimei.2022.100058.
FATTAH, J. et al. Forecasting of demand using ARIMA model. International Journal of Engineering Business Management, v. 10, n. 1, p. 184797901880867, Jan.
2018. Disponível em: https://journals.sagepub.com/doi/10.1177/1847979018808673.
FIORUCCI, J. A. et al. Models for optimising the theta method and their relationship to state space models. International Journal of Forecasting, v. 32, n. 4, p. 1151–
1161, out. 2016. Disponível em: https://doi.org/10.1016/j.ijforecast.2016.02.005.
FOURKIOTIS, K. P.; TSADIRAS, A. Applying Machine Learning and Statistical
Forecasting Methods for Enhancing Pharmaceutical Sales Predictions. Forecasting, v. 6, n. 1, p. 170–186, 1 mar. 2024. Disponível em:
https://doi.org/10.3390/forecast6010010.

---

# Page 65

65
GARDNER, E. S. Exponential smoothing: The state of the art. Journal of
Forecasting, v. 4, n. 1, p. 1–28, 1985. Disponível em:
https://doi.org/10.1002/for.3980040103.
KONTOPOULOU, V. I. et al. A Review of ARIMA vs. Machine Learning Approaches for Time Séries Forecasting in Data Driven Networks. Future Internet, v. 15, n. 8, p.
255, 1 ago. 2023. Disponível em: https://doi.org/10.3390/fi15080255.
LOZIA, Z. Application of modelling and simulation to evaluate the theta method used in diagnostics of automotive shock absorbers. The Archives of Automotive
Engineering – Archiwum Motoryzacji, v. 96, n. 2, p. 5–30, 30 jun. 2022. Disponível em: https://doi.org/10.14669/AM/150823.
MAKRIDAKIS, S.; HIBON, M. The M3-Competition: results, conclusions and implications. International Journal of Forecasting, v. 16, n. 4, p. 451–476, out.
2000. Disponível em: https://doi.org/10.1016/S0169-2070(00)00057-1.
MAKRIDAKIS, S.; WHEELWRIGHT, S. C.; HYNDMAN, R. J. Forecasting: Methods and Applications. In: Elements of Forecasting. Oxfordshire: Taylor & Francis, 1999.
p. 345–346. Disponível em:
https://www.researchgate.net/publication/52008212_Forecasting_Methods_and_Appl ications.
MALIK, Shubham; HARODE, Rohan; KUNWAR, Akash Singh. XGBoost: a deep dive into boosting. Medium Blog, 2020. Disponível em:
http://dx.doi.org/10.13140/RG.2.2.15243.64803.
MCKENZIE, ED. General exponential smoothing and the equivalent arma process. Journal of Forecasting, v. 3, n. 3, p. 333–344, Jul. 1984. Disponível em:
https://doi.org/10.1002/for.3980030312.
MONDAL, P.; SHIT, L.; GOSWAMI, S. Study of Effectiveness of Time Séries
Modeling (Arima) in Forecasting Stock Prices. International Journal of Computer

---

# Page 66

66
Science, Engineering and Applications, v. 4, n. 2, p. 13–29, 30 abr. 2014.
Disponível em: https://doi.org/10.5121/ijcsea.2014.4202.
MURAT, M. et al. Forecasting daily meteorological time séries using ARIMA and regression models. International Agrophysics, v. 32, n. 2, p. 253–264, 1 abr. 2018.
Disponível em: https://doi.org/10.1515/intag-2017-0007.
NEWBOLD, P. ARIMA model building and the time séries analysis approach to forecasting. Journal of Forecasting, v. 2, n. 1, p. 23–35, Jan. 1983. Disponível em:
https://doi.org/10.1002/for.3980020104.
PAO, James J.; SULLIVAN, Danielle S. Time séries sales forecasting. Final year project, Computer Science, Stanford Univ., Stanford, CA, USA, 2017. Disponível em:
https://cs229.stanford.edu/proj2017/final-reports/5244336.pdf.
PARZEN, E. An Approach to Time Séries Analysis. The Annals of Mathematical
Statistics, v. 32, n. 4, p. 951–989, 1961. Disponível em:
https://www.jstor.org/stable/2237900.
SHIRI, F. M. et al. A Comprehensive Overview and Comparative Analysis on Deep
Learning Models. Journal on Artificial Intelligence, v. 6, n. 1, p. 301–360, 2024.
Disponível em: https://doi.org/10.32604/jai.2024.054314.
SPILIOTIS, E.; ASSIMAKOPOULOS, V.; MAKRIDAKIS, S. Generalizing the Theta method for automatic forecasting. European Journal of Operational Research,
Jan. 2020. Disponível em: http://dx.doi.org/10.1016/j.ejor.2020.01.007.
VAVLIAKIS, K.; SIAILIS, A.; SYMEONIDIS, A. Optimizing Sales Forecasting in e-
Commerce with ARIMA and LSTM Models. Proceedings of the 17th International
Conference on Web Information Systems and Technologies, 2021. Disponível em: https://doi.org/10.5220/0010659500003058.