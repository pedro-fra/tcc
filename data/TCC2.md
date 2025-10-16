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

Trabalho

de

Conclusão

de

Curso

apresentado como requisito parcial para

obtenção do título de Bacharel em

Engenharia da Computação, pelo Curso de

Engenharia

da

Computação

da

Universidade do Vale do Rio dos Sinos

## (UNISINOS)

Orientador: Prof. MSc. Jean Schmith

São Leopoldo

2025


---

# Page 3

## RESUMO

Este trabalho tem como objetivo avaliar e comparar o desempenho de

diferentes métodos de previsão de vendas, utilizando tanto técnicas estatísticas

tradicionais quanto algoritmos modernos de aprendizado de máquina, aplicados a

dados reais de faturamento extraídos de um dashboard corporativo em Power BI.

Diante do aumento da competitividade e da demanda por decisões empresariais

baseadas em dados, destaca-se a necessidade de modelos preditivos cada vez mais

precisos e robustos. O estudo envolve a implementação dos modelos ARIMA, Theta,

Suavização Exponencial e XGBoost, analisando suas performances preditivas e as

possibilidades de adoção dessas abordagens no contexto empresarial. Os resultados

são avaliados a partir de métricas estatísticas padronizadas, permitindo identificar se

algum modelo apresenta desempenho superior ao método atualmente empregado. A

pesquisa contribui para a aproximação entre teoria e prática, oferecendo subsídios

para a escolha de métodos de previsão mais adequados às necessidades das

organizações e potencializando o valor estratégico das análises de vendas.

Palavras-chave: Previsão de Vendas; Séries Temporais; Aprendizado de Máquina;

Power BI; ARIMA; XGBoost; Suavização Exponencial; Método Theta; Business

Intelligence.


---

# Page 4

## ABSTRACT

This work aims to evaluate and compare the performance of different sales

forecasting methods, employing both traditional statistical techniques and modern

machine learning algorithms, applied to real revenue data extracted from a corporate

dashboard in Power BI. Given the increasing competitiveness and demand for data-

driven business decisions, there is a growing need for more accurate and robust

predictive models. The study involves the implementation of ARIMA, Theta,

Exponential Smoothing, and XGBoost models, analyzing their predictive performance

and the feasibility of adopting these approaches in corporate environments. The results

are assessed using standardized statistical metrics, allowing for the identification of

models that outperform the currently employed method. This research contributes to

bridging the gap between theory and practice, offering guidance for the selection of

forecasting methods that best fit organizational needs and enhancing the strategic

value of sales analytics.

Key-words: Sales Forecasting; Time Series; Machine Learning; Power BI; ARIMA;

XGBoost; Exponential Smoothing; Theta Method; Business Intelligence.


---

# Page 5

## LISTA DE FIGURAS

Figura 1 - Metodologia geral do trabalho ................................................................... 30

Figura 2 - Metodologia do modelo ARIMA ................................................................ 48

Figura 3 – Metodologia do modelo XGBoost ............................................................. 64

Figura 4 - Metodologia do modelo de Suavização Exponencial ... Erro! Indicador não

definido.

Figura 5 - Metodologia do modelo Theta ...................... Erro! Indicador não definido.


---

# Page 6

6

## LISTA DE QUADROS

Quadro 1 - Cronograma de Desenvolvimento do Projeto ............. Erro! Indicador não

definido.


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

3.1.2 Coleta e integração dos dados ..................................................................... 31

3.1.3 Pré-processamento e transformações dos dados Erro! Indicador não definido.

3.1.4 Análise exploratória e estruturação da série temporal ............................... 38

3.2 MODELOS DE PREVISÃO UTILIZADOS ........................................................... 47

3.2.1 ARIMA .............................................................................................................. 48

3.2.1.1 Importação das bibliotecas e configuração do ambiente ............................... 48

3.2.1.2 Ingestão e conversão dos dados para série temporal ................................... 49

3.2.1.3 Verificação de estacionaridade e diferenciação ............................................ 50

3.2.1.4 Divisão dos dados em conjuntos de treino e teste ........................................ 51

3.2.1.5 Definição dos parâmetros p, d e q ................................................................. 51

3.2.1.6 Treinamento do modelo ................................................................................. 52

3.2.1.7 Validação do modelo e ajustes finos ............................................................. 53

3.2.1.8 Análise residual ............................................................................................. 54

3.2.1.9 Armazenamento dos resultados para comparação futura ............................. 54

3.2.2 XGBoost .......................................................................................................... 64

3.2.2.1 Preparação e engenharia de variáveis .......................................................... 65

3.2.2.2 Divisão dos dados em treino e teste ............................................................. 65

3.2.2.3 Normalização e tratamento dos dados .......................................................... 65

3.2.2.4 Configuração dos hiper parâmetros iniciais ................................................... 65

3.2.2.5 Treinamento inicial do modelo ....................................................................... 66

3.2.2.6 Avaliação inicial de desempenho .................................................................. 67

3.2.2.7 Busca e ajuste de hiper parâmetros .............................................................. 67

3.2.2.8 Validação cruzada e análise de resultados ................................................... 67

3.2.2.9 Geração das previsões finais e armazenamento dos resultados .................. 67

3.2.3 Suavização exponencial .......................................... Erro! Indicador não definido.

3.2.3.1 Preparação dos dados ..................................... Erro! Indicador não definido.

3.2.3.2 Análise exploratória e estrutura da série temporalErro!

Indicador

não

definido.


---

# Page 10

10

3.2.3.3 Divisão em conjunto de treino e teste............... Erro! Indicador não definido.

3.2.3.4 Seleção do tipo de suavização exponencial e parâmetrosErro! Indicador não

definido.

3.2.3.5 Treinamento inicial do modelo .......................... Erro! Indicador não definido.

3.2.3.6 Geração das previsões ..................................... Erro! Indicador não definido.

3.2.3.7 Avaliação do desempenho ............................... Erro! Indicador não definido.

3.2.3.8 Ajuste fino e revalidação .................................. Erro! Indicador não definido.

3.2.3.9 Geração das previsões finais e armazenamento dos resultados ............. Erro!

Indicador não definido.

3.2.4 Theta .......................................................................... Erro! Indicador não definido.

3.2.4.1 Organização e pré-condições dos dados ......... Erro! Indicador não definido.

3.2.4.2 Análise inicial e sazonalidade ........................... Erro! Indicador não definido.

3.2.4.3 Separação temporal para avaliação ................. Erro! Indicador não definido.

3.2.4.4 Configuração e execução do algoritmo ............ Erro! Indicador não definido.

3.2.4.5 Produção das previsões e pós-processamento Erro! Indicador não definido.

3.2.4.6 Avaliação quantitativa e diagnóstico ................ Erro! Indicador não definido.

3.2.4.7 Iteração e consolidação dos resultados ........... Erro! Indicador não definido.

3.3 AVALIAÇÃO E COMPARAÇÃO DOS MODELOS .............................................. 68

3.4 CRONOGRAMA .................................................................................................. 68

REFERÊNCIAS ......................................................................................................... 69


---

# Page 11

11

## 1 INTRODUÇÃO

A previsão de vendas, no contexto atual da transformação digital e da crescente

demanda por decisões empresariais baseadas em dados, se estabelece como um dos

grandes desafios e diferenciais competitivos para organizações de todos os portes.

Com mercados cada vez mais dinâmicos e suscetíveis a variações econômicas,

tecnológicas e comportamentais, a precisão nas estimativas de faturamento assume

papel central no planejamento, controle de estoques, logística, definição de metas e

estratégias comerciais. Este cenário impulsionou o avanço de diferentes métodos de

previsão, desde técnicas estatísticas tradicionais até abordagens inovadoras de

aprendizado de máquina, que vêm transformando a forma como as empresas

analisam e projetam seus resultados futuros.

O uso disseminado de ferramentas de BI, como o Power BI, trouxe grandes

avanços para a visualização e interpretação dos dados históricos das empresas,

permitindo a elaboração de dashboards customizados para acompanhamento do

desempenho de vendas. Contudo, muitos desses sistemas ainda utilizam métodos de

previsão relativamente simples, que podem não captar integralmente a complexidade

dos padrões temporais, sazonalidades e variáveis exógenas presentes nos dados

(ENSAFI et al., 2022). Paralelamente, algoritmos de ML, como o XGBoost, vêm sendo

destacados na literatura por sua elevada acurácia preditiva, robustez e flexibilidade

na incorporação de múltiplos fatores ao processo de modelagem, sendo escolhido

frequentemente em cenários reais e competições internacionais (CHEN; GUESTRIN,

2016).

Diante desse contexto, torna-se pertinente avaliar, sob uma perspectiva

aplicada e comparativa, se modelos de ML podem efetivamente aprimorar as

previsões de faturamento realizadas por soluções já consolidadas no ambiente

empresarial, como o Power BI, contribuindo para a geração de insights mais robustos

e embasados para a tomada de decisão.

## 1.1 TEMA

O presente trabalho aborda o tema da previsão de vendas utilizando séries

temporais, com foco na comparação entre métodos tradicionais e modernos de

modelagem preditiva aplicados a dados reais de faturamento empresarial.


---

# Page 12

12

## 1.2 DELIMITAÇÃO DO TEMA

A pesquisa concentra-se na análise comparativa do desempenho de diferentes

modelos de previsão utilizando dados históricos extraídos de um banco de dados. O

estudo limita-se à previsão de faturamento mensal, simulando o contexto prático

enfrentado por empresas que necessitam estimar o resultado do mês corrente com

base em informações parciais, do primeiro dia do mês até o momento da consulta.

## 1.3 PROBLEMA

O problema que orienta este trabalho é: Modelos avançados de aprendizado

de máquina podem proporcionar previsões mais precisas de faturamento, quando

comparados à abordagem utilizada em dashboards de Power BI? A investigação

busca responder se a adoção de modelos de aprendizado de máquina como XGBoost,

ARIMA, Suavização Exponencial e Theta pode, de fato, melhorar a acurácia das

projeções realizadas atualmente pela empresa, promovendo maior confiabilidade e

valor estratégico às informações disponibilizadas.

## 1.4 OBJETIVOS

1.4.1 Objetivo geral

Avaliar, de forma comparativa, o desempenho de diferentes abordagens de

previsão de vendas, sejam elas tradicionais ou baseadas em ML, aplicadas a dados

reais de faturamento, verificando se algum dos modelos apresenta desempenho

superior ao método atualmente utilizado em dashboards de Power BI.

1.4.2 Objetivos específicos

a) Revisar e contextualizar os principais conceitos de séries temporais,

métodos estatísticos clássicos e técnicas de ML voltadas à previsão de

vendas, conforme descrito por autores como Bezerra (2006), Makridakis,

Wheelwright e Hyndman (1999) e Ensafi et al. (2022);


---

# Page 13

13

b) Estruturar e pré-processar os dados históricos de faturamento de acordo

com as exigências de cada modelo preditivo, assegurando anonimização,

integridade e conformidade com boas práticas de ciência de dados;

c) Implementar, treinar e validar modelos de previsão ARIMA, Theta,

Suavização Exponencial e XGBoost, utilizando métricas estatísticas

padronizadas para avaliação do desempenho;

d) Analisar comparativamente os resultados obtidos e discutir as vantagens,

limitações e possibilidades práticas para adoção dos métodos preditivos no

contexto empresarial.

Acredita-se que essa abordagem possibilite uma análise abrangente e rigorosa,

identificando as oportunidades e desafios envolvidos na transição para modelos

preditivos mais avançados no ambiente corporativo.

## 1.5 JUSTIFICATIVA

A relevância deste estudo se justifica tanto pelo avanço recente das técnicas

de análise preditiva quanto pela necessidade real de organizações aprimorarem seus

processos de tomada de decisão frente a cenários de incerteza e competitividade. Do

ponto de vista acadêmico, há uma lacuna na literatura nacional sobre aplicações

práticas e comparativas de modelos de machine learning em ambientes de BI

amplamente adotados por empresas brasileiras, como o Power BI (ENSAFI et al.,

2022; SHIRI et al., 2024). Internacionalmente, pesquisas vêm demonstrando o

potencial de algoritmos como XGBoost na superação de métodos tradicionais de

previsão, especialmente em séries temporais com padrões complexos e influências

externas (CHEN; GUESTRIN, 2016).

No âmbito empresarial, a adoção de modelos mais precisos pode representar

ganhos substanciais em planejamento, controle financeiro e competitividade,

permitindo que decisões sejam tomadas com maior base quantitativa e menor risco.

Este trabalho, ao propor uma análise comparativa fundamentada, contribui para

aproximar a teoria e a prática, orientando gestores e profissionais de dados quanto à

melhor escolha de métodos para suas demandas específicas.


---

# Page 14

14

## 2 FUNDAMENTAÇÃO TEÓRICA

Neste capítulo, apresenta-se o embasamento teórico indispensável ao

desenvolvimento do presente estudo. Serão discutidos os conceitos fundamentais

relacionados à previsão de dados, contemplando tanto a aplicação de algoritmos de

aprendizado de máquina quanto a utilização de cálculos no Power BI. A partir dessa

fundamentação, busca-se sustentar o estudo de caso realizado, evidenciando as

principais vantagens e limitações de cada abordagem na análise e projeção de

informações.

## 2.1 SÉRIES TEMPORAIS

A análise de séries temporais é uma importante área da estatística, dedicada à

compreensão, modelagem e previsão de fenômenos que são observados de forma

sequencial no tempo. Conforme Bezerra (2006), a utilização da análise de séries

temporais é amplamente difundida em diversas áreas, como economia, meteorologia,

saúde, controle de processos industriais, vendas e finanças, devido à capacidade de

identificar padrões de comportamento e realizar previsões futuras com base em dados

históricos.

2.1.1 Conceitos fundamentais e definições

De acordo com Parzen (1961), uma série temporal pode ser entendida como

um conjunto de observações dispostas cronologicamente, sendo representada

matematicamente como um processo estocástico, no qual cada valor observado

corresponde a um instante específico no tempo.

2.1.2 Características principais

Entre os principais conceitos e características envolvidos na análise de séries

temporais, destacam-se:

a) Estacionariedade: Segundo Bezerra (2006), a estacionariedade ocorre

quando as propriedades estatísticas, tais como média, variância e

covariância, permanecem constantes ao longo do tempo. A condição de


---

# Page 15

15

estacionariedade é importante para aplicação correta de diversos modelos,

como os modelos ARIMA.

b) Tendência: Refere-se à direção predominante da série ao longo do tempo,

podendo ser crescente, decrescente ou estável. Segundo Makridakis,

Wheelwright e Hyndman (1999), a tendência é fundamental para entender o

comportamento das séries e escolher modelos adequados.

c) Sazonalidade: Corresponde às variações periódicas e regulares que

ocorrem em intervalos fixos, como mensal ou anual, devido a fatores

externos

ou

eventos

recorrentes

## (MAKRIDAKIS,

## WHEELWRIGHT;

## HYNDMAN, 1999).

d) Autocorrelação: Representa a correlação da série consigo mesma em

diferentes momentos do tempo (lags). De acordo com Parzen (1961), esse

conceito é fundamental para identificar e compreender o comportamento das

séries temporais.

e) Ruído branco: Para Bezerra (2006), é a parcela aleatória da série

temporal, composta por erros aleatórios independentes com média zero e

variância constante, que não apresentam qualquer tipo de padrão previsível.

2.1.3 Classificações de séries temporais

Makridakis, Wheelwright e Hyndman (1999) classificam as séries temporais em

tipos distintos:

a) Séries estacionárias: Caracterizam-se por apresentar média e variância

constantes ao longo do tempo. São frequentemente observadas em séries

financeiras de retorno.

b) Séries não estacionárias: São séries cujas propriedades estatísticas, como

média e/ou variância, alteram-se com o tempo. Exemplos comuns incluem

séries econômicas como PIB e inflação.

c) Séries lineares e não lineares: Séries lineares podem ser modeladas por

técnicas tradicionais, como ARIMA, enquanto séries não lineares exigem

modelos mais avançados, como redes neurais artificiais (SHIRI et al., 2024).


---

# Page 16

16

2.1.4 Exemplos de aplicação

Vários estudos demonstram a aplicação prática das séries temporais em

diversos contextos:

a) Previsão de vendas no varejo: Ensafi et al. (2022) compararam técnicas

tradicionais como SARIMA e Suavização Exponencial com métodos

avançados como redes neurais LSTM e CNN para previsão das vendas

sazonais de móveis. Os resultados mostraram que as redes neurais LSTM

apresentaram maior precisão na captura de padrões complexos e sazonais.

b) Previsão de vendas semanais em lojas de departamento: Pao e Sullivan

(2014) utilizaram técnicas como árvores de decisão, STL+ARIMA e redes

neurais feed-forward com entradas temporais defasadas, concluindo que as

redes neurais tiveram um desempenho superior, capturando com eficiência

as sazonalidades das vendas semanais.

c) Aplicação de Deep Learning em séries temporais complexas: Shiri et al.

(2024) realizaram uma revisão abrangente sobre o uso de modelos de deep

learning, como CNN, RNN, LSTM e Transformer, em séries temporais. O

estudo apontou que técnicas modernas baseadas em deep learning têm se

mostrado superiores às técnicas tradicionais, principalmente em séries

complexas e com grandes volumes de dados.

## 2.2 MÉTODO THETA

O método Theta ganhou popularidade ao vencer a competição M3 de previsões

de séries temporais devido à sua simplicidade e eficiência em gerar previsões precisas

para diversos tipos de dados. Desde então, este método tem sido amplamente

estudado e aprimorado, resultando em diferentes variantes que exploram seu

potencial para aplicações automáticas e mais robustas. (ASSIMAKOPOULOS;

## NIKILOPOULOS, 2000).


---

# Page 17

17

2.2.1 Descrição geral e origem

O método Theta é uma técnica de previsão uni variada que decompõe a série

temporal original em componentes denominados "linhas Theta". Cada linha Theta é

obtida ajustando-se a curvatura dos dados originais através de um parâmetro θ

aplicado às segundas diferenças da série original. (ASSIMAKOPOULOS;

## NIKILOPOULOS, 2000; SPILIOTIS; ASSIMAKOPOULOS; MAKRIDAKIS, 2020). A

combinação dessas linhas Theta gera previsões que equilibram tendências de curto e

longo prazo. (ASSIMAKOPOULOS; NIKILOPOULOS, 2000).

2.2.2 Fundamentação teórica e parâmetros

As principais características do método Theta incluem:

a) Decomposição da série temporal: a série original é dividida em múltiplas

linhas Theta, destacando diferentes características como tendências de

curto e longo prazo (ASSIMAKOPOULOS; NIKOLOPOULOS, 2000).

b) Parâmetro θ (Theta): controla a curvatura das linhas, com 𝜃< 1 enfatizando

tendências de longo prazo e 𝜃> 1 destacando variações de curto prazo.

## (ASSIMAKOPOULOS;

## NIKOLOPOULOS,

2000;

## SPILIOTIS;

## ASSIMAKOPOULOS; MAKRIDAKIS, 2020).

c) Combinação de previsões: as previsões geradas a partir das linhas Theta

são combinadas usando ponderações específicas para gerar resultados

mais robustos e precisos (FIORUCCI et al., 2016).

d) Flexibilidade e robustez: permite ajuste e adaptação automática dos

parâmetros para diferentes séries temporais, tornando-o versátil para

diversos contextos (SPILIOTIS; ASSIMAKOPOULOS; MAKRIDAKIS, 2020).

e) Eficiência computacional: destaca-se pela simplicidade computacional,

sendo fácil e rápido de implementar, especialmente quando comparado com

métodos mais complexos como ARIMA ou redes neurais (FIORUCCI et al.,

2016).

f) Capacidade de generalização: é aplicável em séries temporais com

diferentes padrões, como tendências lineares, não lineares, séries com


---

# Page 18

18

comportamento

sazonal

e

séries

irregulares

## (SPILIOTIS;

## ASSIMAKOPOULOS; MAKRIDAKIS, 2020).

g) Simplicidade na interpretação: oferece resultados facilmente interpretáveis,

facilitando seu uso prático em ambientes corporativos e industriais

(FIORUCCI et al., 2016).

2.2.3 Equação da linha Theta

Segundo Spiliotis, Assimakopoulos e Makridakis (2020), o método Theta pode

ser matematicamente descrito da seguinte forma:

Seja 𝑌𝑡 uma série temporal observada no tempo 𝑡. Uma linha Theta 𝑍𝑡(𝜃) é

obtida pela expressão:

∇2𝑍𝑡(𝜃) = 𝜃∇2𝑌𝑡= 𝜃(𝑌𝑡−2𝑌(𝑡−1) + 𝑌(𝑡+2)),

𝑡= 3, … , 𝑛

onde ∇2𝑌𝑡 é o operador das segundas diferenças da série original 𝑌 no ponto 𝑡.

2.2.4 Expressões aditivas e multiplicativas

No método Theta, as previsões podem ser realizadas utilizando expressões

aditivas ou multiplicativas:

a) Modelo aditivo: é o modelo original do método Theta, no qual as previsões

são obtidas pela combinação linear aditiva das linhas Theta ajustadas

## (ASSIMAKOPOULOS; NIKOLOPOULOS, 2000).

b) Modelo multiplicativo: é uma extensão recente do método, permitindo

modelar situações em que componentes como sazonalidade e tendência

interagem de forma multiplicativa, sendo especialmente útil em séries com

tendência

exponencial

ou

comportamento

sazonal

multiplicativo

## (SPILIOTIS; ASSIMAKOPOULOS; MAKRIDAKIS, 2020).

2.2.5 Funcionamento do método para previsão de dados futuros

Para prever dados futuros, o método Theta realiza as seguintes etapas

## (ASSIMAKOPOULOS; NIKOLOPOULOS, 2000; FIORUCCI, 2016):


---

# Page 19

19

a) Decomposição: a série temporal é decomposta em linhas Theta com

diferentes curvaturas.

b) Extrapolação: cada linha é extrapolada individualmente, frequentemente

usando métodos simples, como suavização exponencial simples (SES) para

tendências de curto prazo e regressão linear para tendências de longo

prazo.

c) Combinação das linhas: as previsões individuais são combinadas,

geralmente com pesos iguais ou otimizados, produzindo uma previsão final

robusta.

2.2.6 Exemplos práticos de uso

O método Theta tem sido amplamente aplicado em diversas áreas,

demonstrando sua robustez:

a) Competição M3: a versão clássica do método Theta alcançou resultados

superiores às demais técnicas na competição M3, uma famosa competição

internacional focada em métodos de previsão de séries temporais,

especialmente em séries mensais e microeconômicas, destacando-se por

sua precisão e simplicidade (MAKRIDAKIS; HIBON, 2000).

b) Diagnóstico automotivo: Lozia (2022) utilizou o método Theta na avaliação

diagnóstica de amortecedores automotivos, demonstrando a eficácia do

método em modelar e prever o comportamento dinâmico de sistemas

mecânicos complexos.

c) Previsão automática: Spiliotis, Assimakopoulos e Makridakis (2020)

propuseram generalizações do método Theta capazes de selecionar

automaticamente a forma mais apropriada (aditiva ou multiplicativa) e ajustar

a inclinação das tendências, superando outros algoritmos automáticos em

competições recentes (como M4), especialmente em séries anuais.


---

# Page 20

20

## 2.3 MODELO ARIMA

O modelo ARIMA é uma técnica estatística amplamente utilizada para análise

e previsão de séries temporais, desenvolvido por Box e Jenkins (1970). É

especialmente indicado para séries cujos valores passados e erros históricos podem

ser utilizados para prever valores futuros (NEWBOLD, 1983).

2.3.1 Definição e estrutura do modelo ARIMA

O modelo ARIMA é uma combinação dos modelos autorregressivos (AR),

integrados (I) e de médias móveis (MA), definidos pela seguinte notação geral ARIMA

(p, d, q), onde (NEWBOLD, 1983):

a) p: ordem do termo autorregressivo (AR), representa a relação linear entre a

observação atual e as anteriores.

b) d: número de diferenciações necessárias para tornar a série estacionária.

c) q: ordem dos termos de média móvel (MA), que refletem os erros anteriores

do modelo.

Matematicamente, o modelo ARIMA (p, d, q) pode ser expresso da seguinte

forma (NEWBOLD, 1983):

𝑌𝑡=  𝛿 + 𝜙1𝑌𝑡−1 + 𝜙2𝑌𝑡−2 + … + 𝜙𝑝𝑌𝑡−𝑝− 𝜃1𝜀𝑡−1 − 𝜃2𝜀𝑡−2 − … − 𝜃𝑞𝜀𝑡−𝑞+ 𝜀𝑡

Onde:

•

𝑌𝑡: valor atual da série temporal.

•

𝑌𝑡−1, 𝑌𝑡−2,..., 𝑌𝑡−𝑝 : valores anteriores da série temporal (termos AR).

•

𝜀𝑡: erro aleatório (resíduos) com distribuição normal, média zero e variância

constante (ruído branco).

•

𝜀𝑡−1, 𝜀𝑡−2, ..., 𝜀𝑡−𝑞: erros anteriores da série (termos MA).

•

𝛿: constante.

•

𝜙1, 𝜙2, … , 𝜙𝑝: coeficientes do termo autorregressivo.

•

𝜃1, 𝜃2, … , 𝜃𝑞: coeficientes do termo de média móvel.


---

# Page 21

21

2.3.2 Conceitos e características do modelo ARIMA

As principais características do modelo ARIMA incluem (BOX; JENKINS, 1970;

FATTAH et al., 2018):

a) Flexibilidade: Pode ajustar-se a diversas séries temporais, incorporando

tendência, ciclos e sazonalidade.

b) Necessidade de estacionariedade: Séries temporais precisam ser

estacionárias para utilização correta do modelo. A estacionariedade é

geralmente obtida por diferenciação sucessiva das séries temporais.

c) Simplicidade: Fácil de compreender e implementar, apresentando

resultados robustos em previsões de curto prazo.

Para verificar se uma série é estacionária, frequentemente são utilizados testes

estatísticos como o teste Dickey-Fuller (ADF) e o teste KPSS (MURAT et al., 2018).

2.3.3 Como o modelo ARIMA funciona para prever dados futuros?

O processo de construção do modelo ARIMA segue a metodologia Box-

Jenkins, que possui as seguintes etapas (BOX; JENKINS, 1970; MONDAL et al.,

2014):

a) Identificação do modelo: Determinação das ordens p, d e q, com base na

análise gráfica das funções de autocorrelação (ACF) e autocorrelação

parcial (PACF).

b) Estimação dos parâmetros: Os coeficientes do modelo são estimados,

normalmente utilizando o método da máxima verossimilhança.

c) Diagnóstico do modelo: Verificação da adequação do modelo por meio da

análise dos resíduos (erros), usando testes como o teste de Ljung-Box e

critérios estatísticos como AIC (Critério de Informação de Akaike).

d) Previsão: Realização da previsão de valores futuros utilizando o modelo

ajustado.


---

# Page 22

22

2.3.4 Casos práticos e exemplos na literatura

O modelo ARIMA tem diversas aplicações práticas, como evidenciado em

diferentes estudos acadêmicos:

a) Previsão de demanda em indústrias alimentícias: Fattah et al. (2018)

mostraram que o modelo ARIMA (1,0,1) foi eficaz em prever a demanda

futura, ajudando a empresa na gestão eficiente de estoques e redução de

custos.

b) Previsão de vendas no e-commerce: Um modelo híbrido combinando

ARIMA com redes neurais LSTM foi utilizado para previsão precisa em

ambientes com alta volatilidade, como o comércio eletrônico (VAVLIAKIS et

al., 2021).

c) Previsão no mercado farmacêutico: Fourkiotis e Tsadiras (2024) utilizaram

ARIMA em combinação com técnicas de aprendizado de máquina para

prever a demanda por produtos farmacêuticos, mostrando sua eficácia em

capturar efeitos sazonais. Para enfrentar esse desafio, Fourkiotis e Tsadiras

(2024) utilizaram técnicas de análise uni variada de séries temporais para

desenvolver previsões mais precisas. Os autores analisaram uma base de

dados real contendo 600.000 registros históricos de vendas provenientes de

uma farmácia online, abrangendo um período entre 2014 e 2019. A

metodologia proposta envolveu as etapas de pré-processamento e limpeza

de dados, segmentação dos dados, análise exploratória e identificação dos

padrões temporais, aplicação e comparação do modelo ARIMA com

modelos avançados de ML como LSTM e XGBoost e, por fim, avaliação do

modelo com métricas específicas. Os resultados demonstraram que o

modelo ARIMA apresentou uma boa capacidade preditiva ao capturar

adequadamente a sazonalidade e tendências lineares de vendas. Contudo,

os autores destacaram que modelos de ML avançados, especialmente o

XGBoost, tiveram um desempenho ainda superior. Em particular, o XGBoost

obteve as menores taxas de erro absoluto percentual médio (MAPE). Apesar

da boa performance dos modelos avançados de Machine Learning, o

modelo ARIMA ainda obteve desempenho competitivo e foi considerado


---

# Page 23

23

eficaz especialmente em séries temporais com forte componente linear e

sazonalidade bem definida.

d) Previsão de preços no mercado financeiro: Mondal et al. (2014) utilizaram

ARIMA para prever preços de ações, destacando sua simplicidade e

robustez na previsão de tendências.

## 2.4 SUAVIZAÇÃO EXPONENCIAL

O método de suavização exponencial tem recebido grande atenção no contexto

de previsões estatísticas devido à sua eficácia, simplicidade e adaptabilidade na

previsão de séries temporais. Sua popularidade advém da capacidade intrínseca de

atribuir pesos maiores às observações mais recentes em detrimento das observações

mais antigas, permitindo rápidas adaptações às mudanças na dinâmica dos dados

## (GARDNER, 1985; CIPRA, 1992).

Essa técnica tornou-se uma abordagem padrão em diversos campos práticos,

incluindo gestão de estoques, controle de processos industriais, finanças e gestão de

cadeias de suprimentos. Sua ampla adoção se dá pela facilidade computacional e

pela interpretação de suas previsões em comparação com métodos mais complexos

como modelos ARIMA e redes neurais (MCKENZIE, 1984).

2.4.1 Definição e estrutura do método

O método de exponential smoothing é uma técnica recursiva para previsão de

séries temporais que se baseia na ponderação exponencial decrescente das

observações passadas. Formalmente, uma previsão futura é construída como uma

combinação linear entre a observação mais recente e a previsão feita anteriormente.

Essa característica de atualização recursiva confere simplicidade e eficiência

computacional ao método (BROWN, 1962; MCKENZIE, 1984).

Matematicamente, para o SES, a previsão do valor da série temporal 𝑋𝑡+1 pode

ser expressa por:

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

•

𝛼: constante de suavização 0 < 𝛼< 1, que define o grau de ponderação

aplicado ao dado mais recente (BROWN, 1962).

Já métodos mais avançados, como o método de Holt-Winters, consideram

explicitamente os componentes de nível, tendência e sazonalidade da série temporal.

Segundo Gardner (1985), para séries com comportamento sazonal e tendência linear,

a previsão futura para ℎ passos à frente é dada pela expressão geral do método de

Holt-Winters multiplicativo:

𝑋̂𝑡+ℎ= (𝐿𝑡+ ℎ × 𝑏𝑡) ×  𝑆𝑡+ℎ−𝑚(𝑘+1)

Onde:

•

𝐿𝑡 é o nível estimado da série no tempo 𝑡;

•

𝑏𝑡 é a tendência estimada no tempo 𝑡;

•

𝑆𝑡+ℎ−𝑚(𝑘+1) é o fator sazonal estimado no tempo correspondente;

•

ℎ representa o horizonte futuro da previsão (quantidade de períodos à frente);

•

𝑚 é o período sazonal da série (por exemplo, 𝑚= 12 para séries mensais

anuais);

•

𝑘 é o número de ciclos completos transcorridos.

Esses métodos avançados permitem previsões mais precisas em séries

complexas, com tendências claras ou padrões sazonais fortes, superando métodos

mais simples como médias móveis ou o próprio exponential smoothing simples

## (MCKENZIE, 1984; GARDNER, 1985).

2.4.2 Vantagens e limitações na previsão de dados

Entre as características fundamentais do método de exponential smoothing

destacam-se:

a) Adaptabilidade: capacidade de responder rapidamente às alterações

estruturais na série temporal, atribuindo pesos exponenciais aos dados

recentes (GARDNER, 1985).


---

# Page 25

25

b) Simplicidade computacional: a estrutura recursiva dos cálculos torna o

método atrativo em aplicações práticas, especialmente onde é necessária

atualização constante das previsões (BROWN, 1962).

c) Flexibilidade estrutural: diferentes versões, como simples, dupla e tripla

(Holt-Winters), permitem modelar comportamentos como tendência e

sazonalidade com eficiência (MCKENZIE, 1984).

d) Robustez: versões robustas do método, que usam a minimização dos

desvios absolutos ou métodos M-estimadores ao invés de mínimos

quadrados, têm maior resistência a dados atípicos e séries temporais com

distribuições assimétricas ou de caudas pesadas (CIPRA, 1992).

2.4.3 Aplicações e estudos de caso

a) Impacto da suavização exponencial no Efeito Bullwhip: Chen, Ryan e

Simchi-Levi (2000) investigaram como a utilização do exponential smoothing

na previsão de demanda pode intensificar o efeito bullwhip, fenômeno no

qual pequenas variações na demanda são ampliadas ao longo da cadeia de

suprimentos. Eles demonstraram que, ao utilizar previsões com exponential

smoothing, as variações nas demandas observadas pelos fabricantes se

tornam significativamente maiores do que as percebidas pelos varejistas,

aumentando os desafios de gestão e planejamento logístico nas

organizações.

b) Robustez a outliers em séries temporais: Cipra (1992) avaliou o

desempenho de versões robustas do método de exponential smoothing em

séries temporais contaminadas por outliers e distribuições de caudas longas.

Utilizando minimização dos desvios absolutos (norma 𝐿1) em vez dos

mínimos quadrados, Cipra verificou experimentalmente que essas versões

robustas forneceram previsões significativamente mais estáveis e precisas

na presença de valores extremos, superando métodos tradicionais

especialmente em séries financeiras e industriais onde valores atípicos são

comuns.

c) Aplicações em controle de estoques: Gardner (1985) destacou o uso bem-

sucedido de exponential smoothing no controle e previsão para gestão de

estoques. Nesse contexto, foram aplicadas variações do método para prever


---

# Page 26

26

demandas futuras e determinar níveis ótimos de estoque, reduzindo custos

relacionados à manutenção excessiva ou insuficiente de produtos em

inventário. Esse exemplo demonstra claramente como o exponential

smoothing pode auxiliar gestores a otimizarem recursos financeiros e

logísticos nas organizações.

d) Previsões de demanda em séries sazonais e com tendência: McKenzie

(1984) apresentou exemplos práticos demonstrando a eficácia do

exponential smoothing para séries temporais com forte comportamento

sazonal e tendência definida. Em seu estudo, foi utilizado o método Holt-

Winters para capturar esses componentes, proporcionando previsões mais

precisas que outros métodos tradicionais como médias móveis simples e

modelos ARIMA em séries complexas, especialmente no contexto de

demanda sazonal de varejo e setores produtivos.

## 2.5 XGBOOST

O XGBoost tornou-se um dos métodos mais populares e eficazes no âmbito da

previsão e classificação em machine learning, devido à sua capacidade de lidar

eficientemente com grandes quantidades de dados e produzir modelos altamente

precisos. Originalmente proposto por Chen e Guestrin em 2016, o XGBoost combina

otimizações algorítmicas e técnicas avançadas de engenharia de sistemas para

aprimorar significativamente o desempenho de previsões e classificações em diversas

áreas (CHEN; GUESTRIN, 2016).

2.5.1 Visão geral do Extreme Gradient Boosting

O XGBoost é uma implementação otimizada do algoritmo Gradient Boosting,

baseado em árvores de decisão sequenciais. Diferentemente das abordagens

tradicionais, que utilizam árvores independentes (como o Random Forest), o XGBoost

constrói árvores de maneira iterativa, com cada árvore subsequente aprendendo dos

resíduos e erros das anteriores. A combinação final das árvores resulta em um modelo

robusto e altamente eficiente para prever valores futuros e classificar dados

complexos (MALIK; HARODE; KUNWAR, 2020).


---

# Page 27

27

2.5.2 Características e conceitos do XGBoost

Entre as características fundamentais do XGBoost destacam-se:

a) Boosting: Método de aprendizado de máquina que cria um modelo forte por

meio da combinação sequencial de modelos fracos. Cada novo modelo tenta

corrigir os erros dos modelos anteriores (MALIK; HARODE; KUNWAR,

2020).

b) Regularização: O XGBoost incorpora penalidades ao modelo para evitar o

ajuste excessivo (overfitting), limitando a complexidade através de

parâmetros como profundidade máxima das árvores, penalização por

complexidade (gamma) e regularização dos pesos das folhas (lambda).

Essa abordagem resulta em modelos mais generalizáveis (CHEN;

## GUESTRIN, 2016).

c) Sparsity-aware Split Finding: Um algoritmo que otimiza o processo de

divisão das árvores levando em conta a esparsidade dos dados,

economizando recursos computacionais ao ignorar valores ausentes ou

zerados durante a construção das árvores (CHEN; GUESTRIN, 2016).

d) Paralelização e computação distribuída: O XGBoost é projetado para ser

executado em múltiplas CPUs, permitindo o processamento paralelo dos

dados e acelerando significativamente o treinamento de grandes modelos

## (CHEN; GUESTRIN, 2016).

e) Shrinking e Column Subsampling: Técnicas adicionais que ajudam a

controlar a complexidade do modelo. Shrinking reduz o impacto individual

de cada árvore, enquanto Column Subsampling seleciona aleatoriamente

um subconjunto de atributos para cada árvore, aumentando a robustez e a

velocidade do modelo (CHEN; GUESTRIN, 2016).

2.5.3 Como o XGBoost prevê dados futuros

O funcionamento do XGBoost para previsões ocorre de maneira iterativa,

seguindo os passos:


---

# Page 28

28

a) Inicialização: O processo se inicia com a definição de uma previsão inicial,

que geralmente corresponde à média dos valores reais presentes nos dados

de treinamento, no caso de problemas de regressão. Essa previsão inicial

serve como ponto de partida para o modelo e representa a estimativa mais

simples possível sem considerar ainda as relações complexas entre as

variáveis (CHEN; GUESTRIN, 2016; NIELSEN, 2016).

b) Cálculo dos resíduos: Após a obtenção da previsão inicial, calcula-se a

diferença entre os valores previstos e os valores reais, gerando assim os

resíduos. Esses resíduos indicam o quanto o modelo atual está errando na

previsão. O objetivo do XGBoost é reduzir esses resíduos a cada nova

iteração, corrigindo gradualmente as falhas do modelo anterior (NIELSEN,

2016; ZHANG et al., 2021).

c) Treinamento iterativo das árvores: Em cada iteração, uma nova árvore de

decisão é treinada, não para prever diretamente os valores finais, mas sim

para modelar os resíduos obtidos na etapa anterior. Ou seja, cada árvore

seguinte busca aprender e corrigir os erros cometidos pelo conjunto das

árvores anteriores, ajustando-se a padrões ainda não capturados (XIE;

## ZHANG, 2021; NIELSEN, 2016).

d) Atualização das previsões: As previsões do modelo são atualizadas

somando as previsões das novas árvores treinadas às previsões

acumuladas das árvores anteriores. Com isso, o modelo torna-se

progressivamente mais preciso a cada ciclo, pois incorpora sucessivamente

correções dos erros passados. Ao final do processo, a previsão final é

composta pela soma ponderada de todas as árvores criadas durante as

iterações, representando assim uma combinação de múltiplos aprendizados

parciais (CHEN; GUESTRIN, 2016; XIE; ZHANG, 2021).

A função objetivo otimizada no processo é:

𝐿(𝜑) = ∑𝑙(𝑦̂𝑦, 𝑦𝑖)

𝑖

+ ∑Ω(𝑓𝑘)

𝑘

onde:

𝑙(𝑦̂𝑦, 𝑦𝑖) representa a função de perda (e.g., erro quadrático médio);


---

# Page 29

29

Ω(𝑓𝑘) representa o termo de regularização que controla a complexidade

do modelo (CHEN; GUESTRIN, 2016).

2.5.4 Exemplos práticos de uso do XGBoost

a) Utilidades: Segundo Noorunnahar et al. (apud Kontopoulou et al., 2023), no

campo de utilidades, foi conduzido um estudo com o objetivo de prever a

produção anual de arroz em Bangladesh. Os autores compararam a

precisão das previsões feitas por um método ARIMA otimizado,

fundamentado no critério AIC, e pelo algoritmo XGBoost. Para a avaliação

dos modelos, foram consideradas métricas de erro como MAE, MPE, RMSE

e MAPE. Os resultados indicaram que o modelo XGBoost obteve um

desempenho superior em relação ao ARIMA no conjunto de teste,

demonstrando maior eficácia na previsão da produção de arroz para o

contexto analisado.

b) Previsão de volume de vendas no varejo: No setor de utilidades e comércio,

o XGBoost tem se mostrado eficaz na previsão de volumes de vendas. A

pesquisa de Dairu e Shilong (2021) é um exemplo, onde o modelo XGBoost

foi utilizado para prever o volume de vendas no varejo, comparando seus

resultados com o ARIMA clássico, o algoritmo GBDT, um modelo de LSTM

e a ferramenta de previsão Prophet. Os resultados desse estudo indicaram

que as abordagens baseadas em árvores, treinadas com características de

clima e temperatura, ofereceram o melhor desempenho de previsão entre os

cinco modelos, enquanto o ARIMA apresentou o pior desempenho.

Notavelmente, o XGBoost exigiu significativamente menos iterações de

treinamento do que o GBDT e, juntamente com o GBDT, necessitou de

menos dados e recursos em contraste com os modelos de LSTM. Além

disso, os autores propuseram um modelo de previsão de vendas baseado

em XGBoost para um conjunto de dados de bens de varejo do Walmart,

demonstrando bom desempenho com menor tempo de computação e

recursos de memória.


---

# Page 30

30

## 3 METODOLOGIA

Este capítulo apresenta os procedimentos metodológicos adotados para a

realização da presente pesquisa, detalhando de forma sistemática as etapas que

orientaram o desenvolvimento do estudo. São descritos o tipo de pesquisa, a

abordagem utilizada, os métodos de coleta e análise dos dados, bem como os critérios

que fundamentaram as escolhas metodológicas. O objetivo é conferir transparência e

fundamentação científica ao percurso investigativo, garantindo a validade e a

confiabilidade dos resultados obtidos.

## 3.1 METODOLOGIA DE TRABALHO

Com o intuito de proporcionar uma visão geral do percurso metodológico

adotado, a figura a seguir apresenta, de forma esquemática, as principais etapas e

procedimentos desenvolvidos ao longo deste trabalho. O diagrama tem como objetivo

ilustrar, de maneira clara e objetiva, a estrutura metodológica geral que orientou a

condução da pesquisa.

Fonte: elaborado pelo autor

Figura 1 - Metodologia geral do trabalho


---

# Page 31

31

3.1.1 Definição do problema e objetivos da previsão

Este trabalho tem como ponto de partida uma necessidade prática observada

em um dos produtos desenvolvidos pela empresa onde atuo, voltado à análise e

visualização de dados corporativos. Especificamente, trata-se de um dashboard

construído na ferramenta Power BI, que apresenta diversas análises de desempenho,

incluindo uma medida responsável por estimar o faturamento do mês corrente com

base nos dados registrados desde o primeiro dia do mês até o momento da consulta.

O problema que este trabalho propõe a investigar consiste em avaliar se é

possível aprimorar essa estimativa por meio da aplicação de modelos de aprendizado

de máquina. Para isso, serão desenvolvidos diferentes modelos preditivos utilizando

os mesmos dados utilizados atualmente no dashboard, buscando simular o contexto

real de previsão. Em seguida, será avaliado o desempenho de cada modelo com base

em métricas estatísticas, e comparado o resultado mais eficaz com a previsão

atualmente gerada pelo Power BI.

O objetivo principal deste estudo é verificar se algum dos modelos testados

apresenta desempenho superior ao cálculo de previsão utilizado hoje no produto da

empresa. Caso isso ocorra, a adoção do modelo poderá resultar em previsões mais

precisas e na geração de insights mais robustos e estratégicos.

3.1.2 Coleta e pré-processamento dos dados

A coleta e a o pré-processamento dos dados utilizados neste trabalho foram

realizadas através da ferramenta Visual Studio Code. Os dados empregados

correspondem às séries históricas de faturamento disponíveis em um produto interno

da empresa, sendo originalmente utilizados em um dashboard desenvolvido em

Power BI.

Os dados utilizados neste estudo consistiram em registros transacionais de

vendas contendo 37.425 transações no período de 2014 a 2025. Os campos principais

incluíram a data de emissão do pedido, valor líquido da venda, identificação do cliente

e tipo de operação comercial.

O pipeline implementado seguiu uma abordagem sistemática dividida em

etapas distintas, conforme mostra figura abaixo, cada uma com objetivos específicos

para preparar os dados para diferentes tipos de modelos de machine learning.


---

# Page 32

32

Fonte: elaborado pelo autor

3.1.2.1 Criação da variável target

A primeira etapa do pipeline de pré-processamento consistiu na definição e

criação da variável dependente para os modelos preditivos. O processo realizou a

filtragem exclusiva de transações classificadas como "VENDA", excluindo devoluções

e outros tipos de operações comerciais.

O valor líquido das vendas foi então estabelecido como variável target,

representando a quantidade que os modelos tentariam prever. Esta escolha foi

justificada pela relevância direta do valor monetário para decisões de negócio e

planejamento financeiro.

3.1.2.2 Criação de features temporais

A segunda etapa foi a implementação da extração e engenharia de

características temporais a partir da data das transações. Este processo foi

Figura 2 - Metodologia do pré-processamento


---

# Page 33

33

fundamental pois padrões temporais foram cruciais em previsão de vendas,

capturando sazonalidades, tendências e ciclos de negócio.

O sistema extraiu features lineares tradicionais como ano, mês, dia, dia da

semana, trimestre, dia do ano e semana do ano. Estas variáveis capturaram diferentes

granularidades temporais que puderam influenciar o comportamento de vendas.

Adicionalmente, implementou-se codificação trigonométrica (cyclical encoding)

para variáveis temporais cíclicas. Esta técnica matemática utilizou funções seno e

cosseno para representar a natureza circular de variáveis como mês e dia da semana.

Por exemplo, dezembro e janeiro são numericamente distantes (12 e 1) mas

temporalmente adjacentes. A codificação trigonométrica preservou esta proximidade,

permitindo que os modelos compreendessem corretamente as transições cíclicas.

3.1.2.3 Tratamento de valores ausentes

O tratamento de valores ausentes foi implementado através de estratégias

diferenciadas por tipo de dados, reconhecendo que diferentes tipos de variáveis

requereram abordagens distintas.

Para variáveis categóricas, adotou-se o preenchimento com valor constante

"Desconhecido", preservando a informação de ausência como categoria específica.

Esta abordagem evitou a perda de registros e permitiu que os modelos aprendessem

padrões associados à ausência de informação.

Para variáveis numéricas, utilizou-se preenchimento com zero como valor

padrão, considerando que em contexto de vendas, ausência de informação

frequentemente indicou ausência de atividade comercial.

3.1.2.4 Remoção de registros duplicados

A identificação e remoção de duplicatas foi realizada mantendo a primeira

ocorrência de registros idênticos. Esta etapa foi crítica para evitar viés nos modelos

causado por registros repetidos que poderiam inflar artificialmente certas

características dos dados, levando a overfitting e previsões incorretas.

O processo examinou todas as colunas simultaneamente para identificar

registros completamente idênticos, garantindo que apenas duplicatas verdadeiras

fossem removidas, preservando variações legítimas nos dados.


---

# Page 34

34

3.1.2.5 Criação de features agregadas

Esta etapa implementou engenharia de características avançada, criando

features derivadas que capturaram padrões temporais e comportamentais essenciais

para previsão de séries temporais.

As features de lag (defasagem temporal) capturaram dependências históricas

ao incluir valores passados como preditores. Implementaram-se lags de 1, 2, 3, 6 e

12 períodos, permitindo que os modelos identificassem padrões de dependência

temporal em diferentes horizontes. Por exemplo, lag de 12 meses capturou

sazonalidade anual, enquanto lags menores capturaram tendências de curto prazo.

As médias móveis foram calculadas para janelas de 3, 6 e 12 períodos,

suavizando flutuações aleatórias e destacando tendências subjacentes. Estas

features foram particularmente valiosas para modelos de machine learning que

pudessem ter dificuldade em capturar automaticamente padrões temporais

suavizados.

Features agregadas por cliente foram criadas calculando estatísticas

descritivas do comportamento histórico de cada cliente. Estas incluíram valor médio

de compras, desvio padrão (indicando variabilidade do comportamento), frequência

de compras e valor total acumulado. Estas características permitiram que os modelos

personalizassem previsões baseadas no perfil específico de cada cliente.

3.1.2.6 Codificação de variáveis categóricas e processo de anonimização

O processo de codificação foi implementado de forma adaptativa baseada na

cardinalidade das variáveis categóricas, reconhecendo que diferentes técnicas foram

apropriadas para diferentes cenários.

Para variáveis de baixa cardinalidade (até 50 categorias únicas), utilizou-se

One-Hot Encoding, criando variáveis dummy binárias para cada categoria. Esta

abordagem preservou completamente a informação categórica sem impor relações

ordinais artificiais.

Para variáveis de alta cardinalidade (mais de 50 categorias), aplicou-se Label

Encoding, convertendo categorias para valores numéricos ordinais. Esta escolha

equilibrou a preservação de informação com eficiência computacional, evitando


---

# Page 35

35

explosão dimensional que ocorreria com One-Hot Encoding em variáveis muito

categóricas.

O processo de anonimização foi implementado utilizando função de hash

criptográfico MD5 para transformar identificações de clientes em códigos anônimos.

Este processo garantiu três propriedades essenciais: consistência (mesmo cliente

sempre recebeu o mesmo ID anônimo), anonimização irreversível (identidade original

não pôde ser recuperada) e formato padronizado.

O sistema gerou identificadores no formato "CLIENTE_####" onde os quatro

dígitos foram derivados deterministicamente do hash do nome original. Esta

abordagem protegeu a privacidade dos clientes enquanto preservou a capacidade de

análise por cliente individual.

3.1.2.7 Remoção de colunas irrelevantes

Removeram-se colunas que se tornaram redundantes após o processamento,

incluindo a coluna de data original (substituída por features temporais derivadas),

coluna de valor original (substituída pela variável target processada) e coluna de

operação (após filtragem por vendas). Esta limpeza reduziu dimensionalidade e

eliminou informações redundantes que poderiam confundir os algoritmos de

aprendizado.

3.1.2.7 Aplicação de normalização

Implementou-se normalização robusta das variáveis numéricas utilizando

técnica baseada em mediana e quartis ao invés de média e desvio padrão. Esta

escolha foi justificada pela resistência a outliers, particularmente importante em dados

de vendas que frequentemente apresentaram valores extremos devido a transações

excepcionalmente grandes ou pequenas.

A normalização padronizou as escalas das diferentes variáveis, garantindo que

features com magnitudes diferentes contribuíssem equitativamente para o

aprendizado dos modelos. Variáveis como a target e features temporais discretas

foram excluídas da normalização para preservar suas interpretações originais.


---

# Page 36

36

3.1.2.7 Consolidação final dos dados

A etapa final realizou validação e limpeza final dos dados processados.

Qualquer valor ausente remanescente foi tratado através de preenchimento com zero,

garantindo uma base de dados completa para os modelos.

3.1.2.8 Validação de qualidade dos dados

Como filtragem final, removeram-se transações com valores inválidos (zero ou

negativos), garantindo que apenas transações comerciais legítimas fossem utilizadas

no treinamento dos modelos. Este filtro foi aplicado com valor mínimo de 0,01 reais

para eliminar registros potencialmente problemáticos.

3.1.2.9 Saída do processo de pré-processamento

O pipeline de pré-processamento gerou uma base de dados final otimizada

contendo aproximadamente 35.000 transações válidas de venda, mais de 40 variáveis

preditoras, dados mensais agregados cobrindo período do ano de 2014 a 2025,

formato padronizado sem valores ausentes ou duplicatas e variáveis normalizadas

apropriadamente para machine learning.

3.1.2.10 Formatação específica por tipo de modelo

Após a conclusão do pipeline principal de pré-processamento, os dados foram

formatados de maneiras distintas para atender às necessidades específicas de cada

categoria de modelo implementado neste estudo. Esta etapa foi fundamental pois

diferentes algoritmos de machine learning requerem estruturas de dados particulares

para funcionamento otimizado.

3.1.2.10.1 Formato de séries temporais

Para os modelos ARIMA, Theta e Suavização Exponencial, os dados foram

transformados em formato de séries temporais univariadas. Este processo envolveu

a agregação temporal dos dados transacionais em períodos mensais, utilizando a

soma dos valores de vendas como método de agregação.


---

# Page 37

37

O procedimento consistiu em agrupar todas as transações por mês e ano,

calculando o valor total de vendas para cada período mensal. Esta agregação foi

necessária pois os modelos de séries temporais operam com observações

sequenciais regularmente espaçadas no tempo, diferentemente dos dados

transacionais originais que apresentavam múltiplas observações por período.

A série temporal resultante apresentou frequência mensal cobrindo o período

completo dos dados, com cada observação representando o faturamento total do mês

correspondente.

Os dados foram então convertidos para o formato específico da biblioteca

Darts, utilizada para implementação dos modelos de séries temporais. Esta conversão

incluiu a definição adequada do índice temporal e a estruturação dos dados em objeto

TimeSeries compatível com os algoritmos implementados.

3.1.2.10.2 Formato tabular para XGBoost

Para o modelo XGBoost, os dados foram mantidos em formato tabular

expandido, preservando todas as features engenheiradas durante o pré-

processamento. Esta abordagem foi necessária pois algoritmos de gradient boosting,

como o XGBoost, requerem múltiplas variáveis explicativas em formato tabular para

construir árvores de decisão.

A base de dados tabular final conteve 45+ features derivadas das etapas de

pré-processamento, incluindo:

a) Features temporais originais: Ano, mês, dia, trimestre, dia da semana, e

suas respectivas codificações trigonométricas (seno e cosseno) para

capturar padrões cíclicos.

b) Features de dependência temporal: Lags de 1, 2, 3, 6 e 12 períodos que

permitiram ao modelo acessar valores históricos como preditores,

essenciais para capturar dependências temporais em formato tabular.

c) Features de suavização: Médias móveis calculadas para janelas de 3, 6 e

12 períodos, fornecendo versões suavizadas da série que destacam

tendências subjacentes.


---

# Page 38

38

d) Features estatísticas: Medidas de dispersão como desvio padrão, valores

mínimos e máximos calculados em janelas deslizantes, capturando a

variabilidade local dos dados.

e) Features de tendência: Diferenças primeiro-ordem e variações percentuais

que quantificaram mudanças direcionais na série, permitindo ao modelo

identificar padrões de crescimento ou decréscimo.

f) Features comportamentais: Estatísticas agregadas por cliente (média,

desvio padrão, frequência e soma total) que personalizaram as previsões

baseadas no perfil histórico de cada cliente.

g) Features de interação: Combinações multiplicativas entre variáveis

temporais (mês × ano, trimestre × ano) que capturaram efeitos de interação

temporal.

Cada linha da base de dados tabular representou uma observação temporal

com todas as features calculadas para aquele período específico. A variável target foi

mantida como coluna separada, preservando sua escala original para facilitar

interpretação dos resultados.

3.1.3 Análise exploratória e estruturação da série temporal

A análise exploratória de dados (EDA) constitui uma etapa fundamental no

processo de modelagem de séries temporais, precedendo a aplicação de modelos

preditivos e fornecendo insights essenciais sobre a estrutura, padrões e

características dos dados históricos. Conforme destacado por Bezerra (2006), a

compreensão adequada do comportamento temporal dos dados é crucial para a

seleção e parametrização apropriada de modelos de previsão, influenciando

diretamente a qualidade e confiabilidade dos resultados obtidos.

No contexto de séries temporais de vendas, a EDA assume particular

importância devido à complexidade inerente desses dados, que frequentemente

apresentam componentes de tendência, sazonalidade, ciclos econômicos e variações

irregulares. Segundo Makridakis, Wheelwright e Hyndman (1999), a identificação

precisa desses componentes através de técnicas exploratórias adequadas é

fundamental para orientar as decisões metodológicas subsequentes, incluindo a


---

# Page 39

39

escolha de modelos estatísticos apropriados e a definição de estratégias de pré-

processamento.

3.1.3.1 Visão geral da série temporal

A análise exploratória foi implementada através de um sistema automatizado

de visualizações desenvolvido em Python, utilizando bibliotecas especializadas em

análise de séries temporais. Os dados utilizados correspondem à série temporal de

vendas mensais no período de janeiro de 2014 a setembro de 2024, totalizando 133

observações após o pré-processamento e agregação temporal mensal.

A estruturação dos dados seguiu as diretrizes estabelecidas por Parzen (1961),

que define uma série temporal como um conjunto de observações dispostas

cronologicamente, representada matematicamente como um processo estocástico.

Para garantir a adequação dos dados à análise temporal, foi implementada uma

verificação rigorosa da ordenação cronológica, tratamento de valores ausentes e

validação da consistência temporal.

A primeira análise apresenta uma visão geral abrangente da série temporal,

incluindo a evolução das vendas ao longo do tempo com linha de tendência,

distribuição dos valores por ano através de gráficos de boxplot, análise das vendas

acumuladas e volatilidade temporal. Esta visão panorâmica revelou uma tendência de

crescimento consistente de 2014 a 2022, seguida por um declínio significativo entre

os anos 2023 e 2024, com valores variando de aproximadamente R$ 8 milhões em

2014 para um pico de R$ 400 milhões em 2022. A análise de tendência linear mostrou

um coeficiente de determinação (R²) de 0,966, indicando que 96,6% da variação dos

dados é explicada pela tendência temporal.


---

# Page 40

40

Fonte: elaborado pelo autor

3.1.3.2 Decomposição STL

A decomposição STL (Seasonal-Trend using Loess) foi aplicada para separar

os componentes estruturais da série temporal. A decomposição confirmou a presença

de uma tendência de longo prazo bem definida e padrões sazonais consistentes, com

a série original mostrando crescimento exponencial até 2022, seguido por declínio

acentuado. O componente sazonal revelou padrões regulares de variação mensal

com amplitude média de aproximadamente R$ 15 milhões, enquanto o resíduo indicou

períodos de maior volatilidade, especialmente durante os anos de transição

econômica.

Figura 3 - Visão geral da série temporal


---

# Page 41

41

Fonte: elaborado pelo autor

3.1.3.3 Análise de sazonalidade

A análise sazonal detalhada examinou os padrões mensais e de autocorrelação

da série temporal. Foram calculadas as médias mensais históricas, revelando que os

meses de janeiro (R$ 125 milhões), maio (R$ 112 milhões) e dezembro (R$ 118

milhões) apresentam consistentemente os maiores volumes de vendas, enquanto

fevereiro (R$ 87 milhões) e junho (R$ 94 milhões) mostram os menores valores. A

análise de autocorrelação identificou dependências temporais significativas até o lag

12, confirmando a presença de sazonalidade anual na série.

Figura 4 – Decomposição da série temporal


---

# Page 42

42

Fonte: elaborado pelo autor

3.1.3.4 Propriedades estatísticas

A análise das propriedades estatísticas incluiu o cálculo das funções de

autocorrelação (ACF) e autocorrelação parcial (PACF), fundamentais para a

parametrização de modelos ARIMA. A ACF mostrou correlações significativas nos

primeiros lags (0,95 no lag 1), decaindo gradualmente até o lag 12, enquanto a PACF

apresentou cortes abruptos após o primeiro lag (PACF₁ = 0,95, PACF₂ = 0,15),

sugerindo características autorregressivas na série. A análise da série diferenciada

(primeira diferença) confirmou a remoção da tendência, tornando a série mais

adequada para modelagem estatística.

Figura 5 - Análise da sazonalidade


---

# Page 43

43

Fonte: elaborado pelo autor

3.1.3.5 Análise de distribuição

A análise de distribuição dos valores de vendas incluiu histograma com

sobreposição de distribuição normal, gráfico Q-Q para teste de normalidade, box plot

para identificação de outliers, e comparação de densidade. Os resultados indicaram

que a distribuição das vendas não segue uma distribuição normal, apresentando

assimetria positiva (skewness = 1,85) e presença de valores extremos.

Figura 6 - Propriedades estatísticas da série temporal


---

# Page 44

44

Fonte: elaborado pelo autor

3.1.3.6 Evolução temporal detalhada

A análise de evolução temporal examinou as taxas de crescimento anual,

padrões sazonais por ano, e tendência linear geral. O cálculo das taxas de

crescimento revelou crescimento superior a 200% em 2015, estabilização em torno

de 20 a 40% nos anos intermediários, e declínios acentuados de -15% a -48% nos

anos finais. A análise de regressão linear confirmou a equação: Vendas = -2.470.000

× Ano + 5.000.000.000, com R² = 0,966.

Figura 7 - Análise de distribuição


---

# Page 45

45

Fonte: elaborado pelo autor

3.1.3.7 Análise de correlação temporal

A análise de correlação incluiu correlações com lags de 1 a 12 meses,

autocorrelação parcial detalhada, matriz de correlação para lags selecionados e

correlação com componentes temporais (ano, trimestre, mês). Os resultados

mostraram correlações elevadas (>0,8) para os primeiros lags, decaindo

gradualmente até o lag 12. A matriz de correlação dos lags selecionados revelou

padrões de dependência temporal que orientaram a configuração dos modelos

preditivos.

Figura 8 - Evolução temporal das vendas


---

# Page 46

46

Fonte: elaborado pelo autor

3.1.3.8 Insights para modelagem

Com base nesta análise exploratória abrangente, foram identificados os

seguintes resultados fundamentais para a modelagem preditiva:

a) Estacionariedade: A série original não é estacionária devido à forte

tendência, requerendo diferenciação para modelos ARIMA (d = 1);

b) Sazonalidade: Presença confirmada de sazonalidade anual (período 12)

com padrões consistentes;

c) Autocorrelação: Dependências temporais significativas até 12 lags,

orientando a parametrização dos modelos;

d) Distribuição: Dados não seguem distribuição normal;

e) Tendência: Tendência de longo prazo bem definida (R² = 0,966);

Figura 9 - Análise de correlação temporal


---

# Page 47

47

f) Volatilidade: Redução da volatilidade ao longo do tempo, indicando maior

estabilidade nos padrões recentes.

Estes resultados orientaram diretamente a configuração dos parâmetros para

cada modelo preditivo, a escolha das técnicas de pré-processamento específicas, e

as estratégias de validação temporal adotadas nas etapas subsequentes.

## 3.2 MODELOS DE PREVISÃO UTILIZADOS

A modelagem preditiva é a etapa central deste trabalho, sendo responsável por

transformar os dados estruturados em previsões quantitativas para o faturamento do

produto analisado. Considerando as diferentes abordagens e características dos

dados, serão selecionados múltiplos modelos de previsão, cada um com suas próprias

vantagens, desvantagens e requisitos específicos de pré-processamento.

Os modelos escolhidos para este estudo incluem técnicas tradicionais de séries

temporais, como ARIMA e Theta, bem como algoritmos mais recentes e avançados,

como XGBoost, que são amplamente utilizados em aplicações empresariais para

problemas de previsão com séries temporais. Cada um desses modelos foi avaliado

quanto à sua capacidade de capturar padrões históricos, prever tendências futuras e

lidar com os desafios típicos desse tipo de dado, como sazonalidade, tendência e

variações irregulares.

Para garantir uma análise comparativa robusta, foram considerados fatores

como a facilidade de implementação, complexidade computacional e a precisão das

previsões geradas. Além disso, cada modelo será treinado e validado com os mesmos

conjuntos de dados, permitindo uma comparação justa e direta de seu desempenho.

Nos subtópicos a seguir, cada modelo é apresentado individualmente, incluindo

os requisitos específicos para pré-processamento dos dados e o diagrama do fluxo

metodológico correspondente.


---

# Page 48

48

## 3.2.1 ARIMA

A figura a seguir mostra a metodologia utilizada para o modelo.

Fonte: elaborado pelo autor

3.2.1.1 Importação das bibliotecas e configuração do ambiente

A implementação do modelo ARIMA foi realizada utilizando o Visual Studio

Code como ambiente de desenvolvimento integrado, garantindo controle de versão e

reprodutibilidade do código. O ambiente Python foi configurado com as seguintes

bibliotecas essenciais:

Figura 10 - Metodologia do modelo ARIMA


---

# Page 49

49

a) Darts: Biblioteca especializada em séries temporais que forneceu o módulo

AutoARIMA/SARIMA, algoritmos de seleção automática de parâmetros,

métodos de divisão temporal apropriados para séries temporais e funções

integradas de avaliação e diagnóstico.

b) Pandas: Utilizado para manipulação e estruturação inicial dos dados,

conversão de tipos de dados temporais, e operações de agregação e

filtragem durante o pré-processamento.

c) Matplotlib e Seaborn: Empregados para geração de visualizações

diagnósticas, incluindo gráficos de série temporal, correlogramas, análise

de resíduos e comparações entre valores observados e previstos.

Esta preparação foi fundamental para garantir que todas as operações

subsequentes fossem executadas de forma padronizada e rastreável.

3.2.1.2 Ingestão e conversão dos dados para série temporal

O processo de ingestão iniciou com o carregamento dos dados de faturamento

mensal previamente processados na etapa 3.1.2, obtidos do arquivo CSV estruturado

com 133 observações mensais (janeiro 2014 a setembro 2024). Os dados foram

validados quanto à:

a) Integridade temporal: Verificação de continuidade mensal sem lacunas,

confirmação da ordenação cronológica correta, e validação do formato de

datas no padrão ISO (YYYY-MM-DD).

b) Qualidade dos valores: Identificação de valores nulos, negativos ou

extremos que poderiam comprometer a modelagem, e confirmação da

escala monetária consistente (valores em reais).

c) Estrutura adequada: Configuração do índice temporal como DatetimeIndex

do Pandas, garantindo operações temporais apropriadas.

A conversão para o objeto TimeSeries da Darts foi realizada especificando a

coluna de valores (faturamento mensal), o índice temporal (datas mensais), e a

frequência da série ('M' para mensal). Esta estrutura otimizada permitiu que o modelo

ARIMA acessasse funcionalidades avançadas como detecção automática de


---

# Page 50

50

periodicidade sazonal, aplicação de transformações temporais (diferenciação), e

geração de previsões de forma eficiente.

3.2.1.3 Verificação de estacionaridade e diferenciação

A avaliação de estacionariedade foi conduzida considerando os achados da

análise exploratória que evidenciaram forte tendência não linear (crescimento

exponencial até 2022, seguido de declínio acentuado) e padrões sazonais anuais

consistentes.

a) Testes de estacionariedade: Embora o AutoARIMA realize testes internos,

foram realizadas verificações complementares utilizando o teste ADF para

detectar a presença de raiz unitária, e o teste KPSS para confirmar

estacionariedade ao redor de uma tendência determinística.

b) Estratégia de diferenciação: O AutoARIMA foi configurado para explorar

automaticamente:

a. Diferenciação não sazonal (d): Testadas ordens de 0 a 2, sendo d =

1 (primeira diferença) a mais comum para remover tendência linear,

e d = 2 para tendências mais complexas.

b. Diferenciação sazonal (D): Avaliada com período 12 (sazonalidade

anual), testando D = 0 (sem diferenciação sazonal) e D = 1 (uma

diferenciação sazonal para remover padrões sazonais não

estacionários).

O processo de diferenciação foi crucial para transformar a série não

estacionária original em uma série com propriedades estatísticas estáveis, evitando

regressões ilegítimas e garantindo a validade dos pressupostos do modelo ARIMA. A

biblioteca Darts aplicou estas transformações de forma automática e reversível para

as previsões finais.


---

# Page 51

51

3.2.1.4 Divisão dos dados em conjuntos de treino e teste

A divisão temporal foi implementada seguindo rigorosamente o princípio de

não-sobreposição temporal, essencial para validação realística de modelos de séries

temporais. A estratégia adotada foi:

a) Conjunto de treino: Primeiros 107 meses da série (janeiro 2014 a novembro

2022), representando aproximadamente 80% dos dados disponíveis. Este

período incluiu a fase de crescimento consistente e o pico histórico das

vendas, fornecendo ao modelo informação suficiente sobre tendências de

longo prazo e padrões sazonais estabelecidos.

b) Conjunto de teste: Últimos 26 meses da série (dezembro 2022 a setembro

2024), correspondendo a aproximadamente 20% dos dados. Este período

capturou a fase de declínio das vendas, representando um desafio real de

generalização para o modelo.

c) Justificativa da divisão: A proporção 80/20 foi escolhida para garantir

quantidade suficiente de dados para o treinamento (especialmente

importante para capturar múltiplos ciclos sazonais anuais), ao mesmo

tempo que preservou um horizonte de teste representativo para avaliar

performance preditiva em condições adversas.

A implementação utilizou métodos da Darts, que garantiu preservação da

estrutura temporal e evitou vazamento de informações futuras para o conjunto de

treino.

3.2.1.5 Definição dos parâmetros p, d e q

A parametrização do modelo foi conduzida através do AutoARIMA da Darts,

que implementou uma busca sistemática e otimizada pelos melhores parâmetros

SARIMA(p,d,q)(P,D,Q)s. Os parâmetros foram definidos como:


---

# Page 52

52

a) Parâmetros não sazonais:

a. p (ordem autorregressiva): Número de lags da série defasada

utilizados como preditores. Testadas ordens de 0 a 5, onde p = 1

indica dependência do valor anterior, p = 2 inclui os dois valores

anteriores etc.

b. d (ordem de diferenciação): Número de diferenciações aplicadas

para tornar a série estacionária. Avaliadas ordens de 0 a 2, baseadas

nos testes de estacionariedade.

c. q (ordem de média móvel): Número de erros de previsão defasados

incluídos no modelo. Testadas ordens de 0 a 5, capturando

dependências nos termos de erro.

b) Parâmetros sazonais (período s = 12):

a. P (autorregressivo sazonal): Dependência de valores sazonais

defasados (ex.: mesmo mês do ano anterior). Testadas ordens de 0

a 2.

b. D (diferenciação sazonal): Diferenciação aplicada com período

sazonal para remover não estacionariedade sazonal. Avaliadas

ordens de 0 a 1.

c. Q (média móvel sazonal): Erros sazonais defasados incluídos no

modelo. Testadas ordens de 0 a 2.

Para critério de seleção, o AutoARIMA utilizou o AIC para balancear qualidade

do ajuste com parcimônia do modelo, selecionando automaticamente a configuração

que minimizou o AIC. O algoritmo implementou busca stepwise para eficiência

computacional, explorando configurações vizinhas de forma inteligente.

3.2.1.6 Treinamento do modelo

O processo de treinamento foi executado após a seleção automática dos

melhores parâmetros utilizando os algoritmos de estimação implementados na Darts.

O treinamento envolveu:


---

# Page 53

53

a) Estimação por máxima verossimilhança: Os coeficientes do modelo foram

estimados através da maximização da função de verossimilhança, que

encontrou os parâmetros que melhor explicaram os dados observados no

conjunto de treino.

b) Otimização numérica: O processo utilizou algoritmos de otimização não

linear para encontrar os valores ótimos dos coeficientes, iniciando de

valores iniciais estimados e iterando até convergência.

c) Ajuste

da

componente

sazonal:

## O

modelo

## SARIMA

ajustou

simultaneamente os padrões não sazonais (tendência de curto prazo,

dependências de lags próximos) e sazonais (padrões anuais, dependências

de períodos equivalentes em anos anteriores).

d) Validação do ajuste: Durante o treinamento, foram monitoradas métricas de

convergência e estabilidade dos coeficientes estimados para garantir

adequação do processo de otimização.

O resultado foi um modelo completamente parametrizado, capaz de capturar

tanto as dependências temporais de curto prazo quanto os padrões sazonais anuais

identificados na análise exploratória.

3.2.1.7 Validação do modelo e ajustes finos

A etapa de validação consistiu na geração de previsões para todo o horizonte

do conjunto de teste (26 períodos futuros) e avaliação sistemática da performance

preditiva:

a) Geração de previsões: O modelo treinado foi utilizado para produzir

previsões recursivas, onde cada previsão utilizou apenas informações

disponíveis até aquele ponto temporal. Este processo simulou fielmente o

cenário real de previsão operacional.

b) Intervalos de confiança: Foram gerados intervalos de previsão (tipicamente

95% de confiança) baseados na variância estimada dos erros do modelo,

fornecendo medida de incerteza associada a cada previsão.

c) Métricas de avaliação: A performance foi avaliada através do conjunto

padronizado de métricas:


---

# Page 54

54

a. MAE: Erro absoluto médio em reais, interpretável diretamente na

escala do problema.

b. RMSE: Raiz do erro quadrático médio, penalizando mais fortemente

grandes desvios.

c. MAPE: Erro percentual absoluto médio, permitindo interpretação

relativa independente da escala.

d. R²: Coeficiente de determinação, medindo proporção da variância

explicada pelo modelo.

e. Acurácia Direcional: Proporção de acertos na direção de variação

(crescimento/decrescimento) entre períodos consecutivos.

d) Análise temporal das previsões: Foi conduzida análise período a período

para identificar padrões nos erros, sazonalidade residual, e performance

diferencial ao longo do horizonte de previsão.

3.2.1.8 Análise residual

Uma análise detalhada dos resíduos do modelo foi conduzida para verificar se

os erros de previsão se distribuíram de forma aleatória, sem padrões sistemáticos não

modelados. Foram gerados gráficos de autocorrelação (ACF) e autocorrelação parcial

(PACF) dos resíduos, buscando confirmar comportamento próximo ao ruído branco.

Resíduos com padrões significativos indicaram que o modelo não conseguiu

capturar completamente as relações temporais nos dados. Adicionalmente, a análise

incluiu inspeção visual da distribuição dos resíduos e identificação de outliers ou

eventos atípicos que poderiam comprometer a precisão das previsões futuras. Esta

validação foi essencial para confirmar a adequação do modelo selecionado.

3.2.1.9 Armazenamento dos resultados para comparação futura

Foram geradas visualizações específicas para documentar o desempenho do

modelo ARIMA, incluindo gráficos de série temporal comparando valores observados

e previstos, análise de resíduos ao longo do tempo e representação gráfica da

estrutura de correlação do conjunto de dados para diagnóstico.


---

# Page 55

55

Os resultados do modelo ARIMA, incluindo previsões, métricas de

desempenho, parâmetros selecionados e diagnósticos foram salvos de forma

estruturada para posterior comparação com os demais modelos (Theta, Suavização

Exponencial e XGBoost) e com as previsões atualmente utilizadas no Power BI. Esta

documentação foi essencial para a análise comparativa final e escolha da abordagem

preditiva mais adequada ao contexto empresarial.

3.2.2 Suavização Exponencial

A figura a seguir mostra a metodologia utilizada para o modelo.

Fonte: elaborado pelo autor

Figura 11 – Metodologia do modelo Suavização Exponencial


---

# Page 56

56

O modelo de Suavização Exponencial compartilhou grande parte da

metodologia com o ARIMA, diferindo principalmente na abordagem de modelagem e

nos critérios de seleção do modelo. As etapas de importação de bibliotecas, ingestão

e conversão de dados e a divisão treino/teste foram executadas de forma idêntica ao

ARIMA, utilizando a mesma biblioteca Darts, mesma estrutura TimeSeries, e mesma

proporção 80/20 com divisão temporal rigorosa.

3.2.2.1 Análise de componentes para seleção do modelo

Diferentemente do ARIMA, que se baseou em testes de estacionariedade e

análise de correlogramas, o modelo de Suavização Exponencial utilizou os resultados

da decomposição STL já realizada na análise exploratória para orientar a seleção do

tipo apropriado de modelo.

Com base nos componentes já extraídos na EDA, foram calculadas métricas

quantitativas específicas para Suavização Exponencial:

a) Força da tendência: Este cálculo utilizou os componentes da decomposição

STL previamente realizada.

b) Força da sazonalidade: Novamente utilizando os resultados da EDA.

c) Lógica de seleção automática: A biblioteca Darts implementou critérios

automáticos para escolha entre:

a. Suavização Exponencial Simples (SES): Para séries sem tendência

ou sazonalidade significativas

b. Método de Holt: Para séries com tendência forte, mas sazonalidade

fraca

c. Método de Holt-Winters: Para séries com ambos os componentes

significativos (caso esperado desta série)


---

# Page 57

57

3.2.2.2 Decisão entre modelo aditivo e multiplicativo

Uma etapa específica da Suavização Exponencial foi a escolha entre formulações

aditiva e multiplicativa, baseada na análise dos componentes sazonais da EDA:

a) Modelo Aditivo: Selecionado quando a amplitude da sazonalidade

permaneceu relativamente constante ao longo do tempo.

b) Modelo Multiplicativo: Selecionado quando a amplitude da sazonalidade

variou proporcionalmente ao nível da série.

A decisão foi automatizada pela Darts baseada na análise da variância relativa

dos componentes sazonais já extraídos na EDA, evitando recompilação

desnecessária.

3.2.2.3 Configuração e otimização de parâmetros

Ao contrário do ARIMA, que utilizou parâmetros discretos (p, d, q), a Suavização

Exponencial otimizou parâmetros contínuos de suavização:

a) Parâmetros do modelo Holt-Winters:

a. α (alfa): Parâmetro de suavização do nível (0 < α ≤ 1)

b. β (beta): Parâmetro de suavização da tendência (0 ≤ β ≤ 1)

c. γ (gama): Parâmetro de suavização sazonal (0 ≤ γ ≤ 1)

b) Período sazonal: Fixado em 12 meses conforme evidenciado na EDA

c) Processo de otimização: A Darts utilizou algoritmos de minimização

numérica para encontrar os valores ótimos que minimizaram o erro

quadrático médio no conjunto de treino, diferindo do critério AIC usado no

## ARIMA.


---

# Page 58

58

3.2.2.4 Treinamento por suavização recursiva

## O

processo

de

treinamento

diferiu

fundamentalmente

do

## ARIMA

por

utilizar suavização exponencial recursiva ao invés de estimação de máxima

verossimilhança:

a) Inicialização dos componentes:

a. Nível inicial: Estimado como média dos primeiros períodos

b. Tendência inicial: Calculada como diferença média inicial

b) Índices sazonais: Estimados através dos primeiros ciclos da série

c) Atualização recursiva: Para cada período t do treino, os componentes foram

atualizados:

a. Nível suavizado através de combinação ponderada do valor

observado e nível anterior projetado

b. Tendência suavizada através de combinação da diferença de nível

recente e tendência anterior

c. Índice sazonal atualizado com base no desvio sazonal observado

Este processo iterativo permitiu ao modelo adaptar-se gradualmente aos

padrões, diferindo da estimação simultânea de todos os parâmetros no ARIMA.

3.2.2.5 Geração de previsões diretas

A geração de previsões na Suavização Exponencial utilizou abordagem direta (não

recursiva) baseada nos componentes finais:

1. Mecânica de previsão: Para cada horizonte h:

a. Nível futuro projetado adicionando tendência × h ao último nível

b. Componente sazonal obtido do índice correspondente ao período do

ano

c. Previsão final através de combinação aditiva ou multiplicativa


---

# Page 59

59

Esta abordagem diferiu das previsões recursivas do ARIMA, sendo mais

apropriada para modelos de suavização.

3.2.2.6 Análise residual específica para suavização

A análise residual seguiu protocolo similar ao ARIMA (seção 3.2.1.8), mas com focos

específicos:

a) Validação de componentes: Além da análise de aleatoriedade dos resíduos,

foi verificada a tendência e sazonalidade.

b) Estabilidade dos parâmetros: Foram analisados os valores otimizados de α,

β e γ para confirmar estabilidade numérica (valores não próximos aos limites

0 ou 1, que indicariam problemas de convergência).

c) Adequação do modelo selecionado: Foi confirmada a escolha entre

aditivo/multiplicativo através de análise visual dos resíduos padronizados e

métricas de ajuste.

3.2.3 Theta

O modelo Theta compartilhou as etapas fundamentais de preparação com os

modelos anteriores, diferindo principalmente na abordagem de decomposição e

extrapolação. As etapas de importação de bibliotecas, ingestão e conversão de dados

e divisão treino/teste foram executadas de forma idêntica ao ARIMA, utilizando a

mesma biblioteca Darts, mesma estrutura TimeSeries, e mesma divisão temporal

80/20.

A figura a seguir mostra a metodologia utilizada para o modelo.


---

# Page 60

60

Fonte: elaborado pelo autor

3.2.3.1 Verificação de pré-condições do método Theta

O método Theta na biblioteca Darts exigiu verificações específicas antes da aplicação,

diferindo dos modelos anteriores:

a) Validação da série temporal: Foi confirmada a ausência de valores nulos na

série, pois o Theta da Darts não possui tratamento automático para dados

Figura 12 – Metodologia do modelo Theta


---

# Page 61

61

ausentes, diferentemente do ARIMA que pode interpolar valores durante o

ajuste.

b) Verificação de univariância: O método foi aplicado exclusivamente à série

temporal univariada de faturamento mensal, sem variáveis explicativas

adicionais,

seguindo

a

natureza

original

do

método

proposto

por Assimakopoulos e Nikolopoulos (2000).

c) Confirmação de regularidade temporal: Foi verificada a frequência mensal

constante da série (133 observações consecutivas), requisito para a

decomposição Theta funcionar adequadamente.

3.2.3.2 Configuração automática vs. manual do modelo

O método Theta da Darts ofereceu configuração totalmente automática:

a) Parâmetro Theta (θ): A Darts implementou seleção automática do parâmetro

θ, que controla a curvatura das linhas Theta. Valores θ < 1 enfatizam

tendências de longo prazo, enquanto θ > 1 destacam variações de curto

prazo, conforme Spiliotis, Assimakopoulos e Makridakis (2020).

b) Detecção automática de sazonalidade: O Theta detectou automaticamente

a presença e o período da sazonalidade (12 meses) com base nos padrões

da série.

c) Configuração de decomposição: O modelo foi configurado para aplicar

decomposição automática da série em componentes Theta, sem

necessidade de especificação manual de ordens ou tipos de componentes.

3.2.3.3 Decomposição e criação das linhas Theta

Esta etapa foi específica do método Theta e diferiu fundamentalmente dos outros

modelos:


---

# Page 62

62

a) Aplicação das segundas diferenças: O método aplicou o operador de

segundas diferenças à série original conforme a formulação matemática

de Assimakopoulos e Nikolopoulos (2000).

b) Geração das linhas Theta: Foram criadas múltiplas linhas Theta através de

transformações matemáticas.

c) Extração de componentes: O processo extraiu automaticamente:

a. Linha Theta 0 (θ = 0): Representa tendência linear de longo prazo

b. Linha Theta 2 (θ = 2): Captura variações de curto prazo e

sazonalidade

c. Linhas

intermediárias:

Quando

aplicável,

para

capturar

características específicas da série

3.2.3.4 Treinamento e ajuste das componentes

O processo de treinamento do Theta diferiu dos modelos de suavização exponencial

e ARIMA:

a) Ajuste das linhas individuais: Cada linha Theta foi ajustada separadamente

utilizando métodos apropriados:

a. Linha Theta 0: Ajustada por regressão linear para capturar tendência

de longo prazo

b. Linha Theta 2: Ajustada por Suavização Exponencial Simples (SES)

para variações de curto prazo

b) Otimização automática: A Darts implementou otimização automática dos

parâmetros de cada componente, incluindo constantes de suavização para

as linhas de curto prazo e coeficientes de tendência para linhas de longo

prazo.

c) Validação da decomposição: O processo verificou a adequação da

decomposição através de análise dos componentes extraídos e sua

capacidade de reconstruir a série original.


---

# Page 63

63

3.2.3.5 Combinação de previsões e extrapolação

A geração de previsões seguiu abordagem única de combinação de extrapolações:

a) Extrapolação individual: Cada linha Theta foi extrapolada separadamente

para o horizonte de teste:

a. Tendência de longo prazo: Extrapolada linearmente baseada na

linha Theta 0

b. Componente de curto prazo: Extrapolada através do último nível

suavizado da linha Theta 2

b) Combinação ponderada: As previsões finais foram obtidas através de

combinação das extrapolações individuais, tipicamente com pesos iguais

(0,5 para cada componente) ou pesos otimizados baseados na performance

histórica, seguindo Fiorucci et al. (2016).

c) Tratamento de sazonalidade: Quando presente, a sazonalidade foi

incorporada através da extrapolação da linha Theta 2, que capturou padrões

de curto prazo incluindo variações sazonais.

3.2.3.6 Avaliação e diagnósticos específicos

A avaliação seguiu protocolo similar aos modelos anteriores, com análises

específicas:

a) Validação das linhas Theta: Foi verificada a adequação da decomposição

através de:

a. Análise da suavidade das linhas extraídas

b. Verificação da capacidade de reconstrução da série original

c. Avaliação da interpretabilidade das componentes (tendência vs.

variações)


---

# Page 64

64

d. Análise de estabilidade: Foram examinados os parâmetros

otimizados de cada linha para confirmar convergência e estabilidade

numérica.

3.2.4 XGBoost

A figura 3 mostra a metodologia utilizada para o modelo.

Fonte: elaborado pelo autor

Figura 13 – Metodologia do modelo XGBoost


---

# Page 65

65

3.2.4.1 Preparação e engenharia de variáveis

Diferentemente do ARIMA, cuja entrada é a própria série temporal univariada,

o XGBoost exige que a série seja transformada em uma base tabular. Serão criadas

variáveis defasadas, médias móveis e estatísticas que descrevam a série ao longo do

tempo. Além disso, poderão ser adicionadas variáveis de calendário (mês, dia da

semana, feriados etc.), enriquecendo o conjunto de treinamento com informações

contextuais. Esta etapa é exclusiva e essencial para o XGBoost, pois permite ao

modelo explorar dependências temporais e efeitos sazonais/exógenos.

3.2.4.2 Divisão dos dados em treino e teste

Assim como no ARIMA, os dados serão divididos em conjuntos de treino e

teste, sempre respeitando a ordem cronológica para evitar vazamento de informações

futuras.

3.2.4.3 Normalização e tratamento dos dados

Esta etapa, embora similar à limpeza realizada no ARIMA, será orientada para

o contexto tabular. Serão tratados valores ausentes gerados na criação de lags e

médias móveis por meio de imputação ou exclusão. Se necessário, as variáveis

poderão ser normalizadas ou padronizadas para garantir melhor desempenho do

algoritmo.

3.2.4.4 Configuração dos hiper parâmetros iniciais

Diferentemente do ARIMA, em que os parâmetros de configuração são (p, d, q)

definidos com base em análise de autocorrelação da própria série temporal, o modelo

XGBoost depende de um conjunto mais amplo de hiper parâmetros que controlam

tanto a complexidade quanto o desempenho do algoritmo de árvores de decisão.

Entre os principais hiper parâmetros que deverão ser configurados inicialmente,

destacam-se:

a) n_estimators (número de árvores): Define quantas árvores de decisão serão

criadas e combinadas pelo modelo.


---

# Page 66

66

b) max_depth (profundidade máxima): Limita a quantidade de divisões que

cada árvore pode fazer, afetando a capacidade de capturar padrões

complexos sem sobre ajuste.

c) learning_rate (taxa de aprendizado): Controla o peso de cada nova árvore

adicionada no processo de boosting, influenciando diretamente a velocidade

e a estabilidade do treinamento.

d) subsample (amostragem): Determina a fração de exemplos utilizados para

treinar cada árvore, o que pode aumentar a generalização do modelo.

e) colsample_bytree: Define a proporção de variáveis consideradas em cada

divisão, reduzindo a chance de sobre ajuste.

A seleção inicial desses hiper parâmetros poderão ser realizadas com base em

estudos prévios, valores sugeridos na literatura ou ainda com valores padrão do

próprio XGBoost. É importante salientar que, diferentemente do ARIMA, o XGBoost

permite grande flexibilidade na escolha e combinação desses hiper parâmetros,

tornando o processo de ajuste potencialmente mais complexo e exigente em termos

de experimentação.

3.2.4.5 Treinamento inicial do modelo

O processo de treinamento inicial do XGBoost se diferencia substancialmente

do ARIMA, principalmente pela estrutura dos dados e pelo mecanismo de

aprendizado.

Enquanto o ARIMA utiliza uma série temporal univariada e ajusta seus

parâmetros para capturar padrões autorregressivos e de média móvel, o XGBoost irá

trabalhar sobre uma base tabular composta por múltiplas features, incluindo variáveis

defasadas (lags), médias móveis, variáveis sazonais e de calendário, entre outras. O

modelo será treinado utilizando o conjunto de treino previamente definido, buscando

construir sucessivas árvores de decisão (de acordo com o número definido em

n_estimators) que, em conjunto, minimizarão o erro de previsão.

Durante esse processo, cada nova árvore será construída para corrigir os erros

cometidos pelas árvores anteriores, em um procedimento iterativo chamado boosting.

O ajuste do modelo será realizado até que todos os dados de treino tenham sido


---

# Page 67

67

utilizados para aprender os padrões relevantes da série temporal e de suas variáveis

derivadas.

Ao final do treinamento inicial, o modelo estará preparado para realizar

previsões sobre o conjunto de teste, e os resultados obtidos servirão como base para

a avaliação inicial de desempenho e para eventuais ajustes de hiper parâmetros em

etapas subsequentes.

3.2.4.6 Avaliação inicial de desempenho

A avaliação do desempenho inicial será realizada de maneira análoga ao

ARIMA, por meio de métricas como RMSE, MAE ou MAPE, aplicadas ao conjunto de

teste. A análise dos erros também poderá indicar a necessidade de ajuste nas features

ou nos hiper parâmetros.

3.2.4.7 Busca e ajuste de hiper parâmetros

Enquanto o ajuste de parâmetros do ARIMA envolve os valores de p, d, q, no

XGBoost será realizada uma busca sistemática para identificar os melhores hiper

parâmetros do modelo, como taxa de aprendizado, número de árvores e profundidade

máxima.

3.2.4.8 Validação cruzada e análise de resultados

Assim como no ARIMA, será empregada validação cruzada adequada a séries

temporais, assegurando a robustez dos resultados e a ausência de sobre ajuste. Os

resultados da validação serão analisados quanto à consistência e possíveis padrões

residuais.

3.2.4.9 Geração das previsões finais e armazenamento dos resultados

Por fim, as previsões finais geradas pelo modelo XGBoost serão armazenadas

para comparação direta com os resultados do ARIMA, dos demais modelos avaliados

e com as previsões atualmente geradas pelo Power BI.


---

# Page 68

68

## 3.3 AVALIAÇÃO E COMPARAÇÃO DOS MODELOS

Após o ajuste e validação de todos os modelos preditivos considerados neste

trabalho, será realizada uma comparação quantitativa do desempenho de cada

modelo utilizando as seguintes métricas estatísticas, recomendadas pela literatura

para problemas de previsão de séries temporais:

a) Erro Médio Absoluto (MAE);

b) Raiz do Erro Quadrático Médio (RMSE);

c) Erro Percentual Absoluto Médio (MAPE).

Essas métricas serão calculadas para o conjunto de teste de cada modelo. O

modelo que apresentar o menor valor de erro (considerando principalmente MAE e

RMSE), será selecionado como o modelo de melhor desempenho, conforme

abordagem utilizada por Hyndman et al. (1999) e Gardner (1985).

Na sequência, o modelo de melhor desempenho será comparado diretamente

ao método de previsão atualmente empregado no Power BI. Essa comparação será

realizada utilizando as mesmas métricas, com o objetivo de identificar se a abordagem

baseada em aprendizado de máquina ou métodos estatísticos apresenta ganhos

significativos de acurácia em relação à solução já adotada no produto da empresa.

A escolha final do modelo será baseada não apenas no desempenho

quantitativo, mas também na sua viabilidade de implementação e integração à

plataforma existente, conforme recomendam Gardner (1985) e Hyndman et al. (1999).


---

# Page 69

69

## REFERÊNCIAS

ASSIMAKOPOULOS, V.; NIKOLOPOULOS, K. The Theta model: a decomposition

approach to forecasting. International Journal of Forecasting, v. 16, n. 4, p. 521–

530, out. 2000. Disponível em: https://doi.org/10.1016/S0169-2070(00)00066-2.

BEZERRA, Manoel Ivanildo Silvestre. Apostila de Análise de Séries Temporais.

São Paulo: UNESP, 2006. Disponível em:

https://www.ibilce.unesp.br/Home/Departamentos/MatematicaEstatistica/apostila_ser

ies_temporais_unesp.pdf.

BOX, G. E. P. et al. Time series analysis: forecasting and control. Hoboken, New

Jersey: John Wiley & Sons, 2015.

CHEN, T.; GUESTRIN, C. XGBoost: a Scalable Tree Boosting System. Proceedings

of the 22nd ACM SIGKDD International Conference on Knowledge Discovery

and Data Mining - KDD ’16, v. 1, n. 1, p. 785–794, 13 ago. 2016. Disponível em:

https://doi.org/10.1145/2939672.2939785.


---

# Page 70

70

DAIRU, X.; SHILONG, Z. Machine Learning Model for Sales Forecasting by

Using XGBoost. Disponível em:

https://doi.org/10.1109/ICCECE51280.2021.9342304.

ENSAFI, Y. et al. Time-series forecasting of seasonal items sales using machine

learning – A comparative analysis. International Journal of Information

Management Data Insights, v. 2, n. 1, p. 100058, abr. 2022. Disponível em:

https://doi.org/10.1016/j.jjimei.2022.100058.

FATTAH, J. et al. Forecasting of demand using ARIMA model. International Journal

of Engineering Business Management, v. 10, n. 1, p. 184797901880867, jan.

2018. Disponível em: https://journals.sagepub.com/doi/10.1177/1847979018808673.

FIORUCCI, J. A. et al. Models for optimising the theta method and their relationship

to state space models. International Journal of Forecasting, v. 32, n. 4, p. 1151–

1161, out. 2016. Disponível em: https://doi.org/10.1016/j.ijforecast.2016.02.005.

FOURKIOTIS, K. P.; TSADIRAS, A. Applying Machine Learning and Statistical

Forecasting Methods for Enhancing Pharmaceutical Sales Predictions. Forecasting,

v. 6, n. 1, p. 170–186, 1 mar. 2024. Disponível em:

https://doi.org/10.3390/forecast6010010.

GARDNER, E. S. Exponential smoothing: The state of the art. Journal of

Forecasting, v. 4, n. 1, p. 1–28, 1985. Disponível em:

https://doi.org/10.1002/for.3980040103.

KONTOPOULOU, V. I. et al. A Review of ARIMA vs. Machine Learning Approaches

for Time Series Forecasting in Data Driven Networks. Future Internet, v. 15, n. 8, p.

255, 1 ago. 2023. Disponível em: https://doi.org/10.3390/fi15080255.

LOZIA, Z. Application of modelling and simulation to evaluate the theta method used

in diagnostics of automotive shock absorbers. The Archives of Automotive


---

# Page 71

71

Engineering – Archiwum Motoryzacji, v. 96, n. 2, p. 5–30, 30 jun. 2022. Disponível

em: https://doi.org/10.14669/AM/150823.

MAKRIDAKIS, S.; HIBON, M. The M3-Competition: results, conclusions and

implications. International Journal of Forecasting, v. 16, n. 4, p. 451–476, out.

2000. Disponível em: https://doi.org/10.1016/S0169-2070(00)00057-1.

MAKRIDAKIS, S.; WHEELWRIGHT, S. C.; HYNDMAN, R. J. Forecasting: Methods

and Applications. In: Elements of Forecasting. Oxfordshire: Taylor & Francis, 1999.

p. 345–346. Disponível em:

https://www.researchgate.net/publication/52008212_Forecasting_Methods_and_Appl

ications.

MALIK, Shubham; HARODE, Rohan; KUNWAR, Akash Singh. XGBoost: a deep

dive into boosting. Medium Blog, 2020. Disponível em:

http://dx.doi.org/10.13140/RG.2.2.15243.64803.

MCKENZIE, ED. General exponential smoothing and the equivalent arma

process. Journal of Forecasting, v. 3, n. 3, p. 333–344, jul. 1984. Disponível em:

https://doi.org/10.1002/for.3980030312.

MONDAL, P.; SHIT, L.; GOSWAMI, S. Study of Effectiveness of Time Series

Modeling (Arima) in Forecasting Stock Prices. International Journal of Computer

Science, Engineering and Applications, v. 4, n. 2, p. 13–29, 30 abr. 2014.

Disponível em: https://doi.org/10.5121/ijcsea.2014.4202.

MURAT, M. et al. Forecasting daily meteorological time series using ARIMA and

regression models. International Agrophysics, v. 32, n. 2, p. 253–264, 1 abr. 2018.

Disponível em: https://doi.org/10.1515/intag-2017-0007.

NEWBOLD, P. ARIMA model building and the time series analysis approach to

forecasting. Journal of Forecasting, v. 2, n. 1, p. 23–35, jan. 1983. Disponível em:

https://doi.org/10.1002/for.3980020104.


---

# Page 72

72

PAO, James J.; SULLIVAN, Danielle S. Time series sales forecasting. Final year

project, Computer Science, Stanford Univ., Stanford, CA, USA, 2017. Disponível em:

https://cs229.stanford.edu/proj2017/final-reports/5244336.pdf.

PARZEN, E. An Approach to Time Series Analysis. The Annals of Mathematical

Statistics, v. 32, n. 4, p. 951–989, 1961. Disponível em:

https://www.jstor.org/stable/2237900.

SHIRI, F. M. et al. A Comprehensive Overview and Comparative Analysis on Deep

Learning Models. Journal on Artificial Intelligence, v. 6, n. 1, p. 301–360, 2024.

Disponível em: https://doi.org/10.32604/jai.2024.054314.

SPILIOTIS, E.; ASSIMAKOPOULOS, V.; MAKRIDAKIS, S. Generalizing the Theta

method for automatic forecasting. European Journal of Operational Research, jan.

2020. Disponível em: http://dx.doi.org/10.1016/j.ejor.2020.01.007.

VAVLIAKIS, K.; SIAILIS, A.; SYMEONIDIS, A. Optimizing Sales Forecasting in e-

Commerce with ARIMA and LSTM Models. Proceedings of the 17th International

Conference on Web Information Systems and Technologies, 2021. Disponível

em: https://doi.org/10.5220/0010659500003058.
