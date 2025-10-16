# 3 METODOLOGIA

Este capítulo apresenta os procedimentos metodológicos adotados para a realização da presente pesquisa, detalhando de forma sistemática as etapas que orientaram o desenvolvimento do estudo. São descritos o tipo de pesquisa, a abordagem utilizada, os métodos de coleta e análise dos dados, bem como os critérios que fundamentaram as escolhas metodológicas. O objetivo é conferir transparência e fundamentação científica ao percurso investigativo, garantindo a validade e a confiabilidade dos resultados obtidos.

## 3.1 METODOLOGIA DE TRABALHO

Com o intuito de proporcionar uma visão geral do percurso metodológico adotado, a figura a seguir apresenta, de forma esquemática, as principais etapas e procedimentos desenvolvidos ao longo deste trabalho. O diagrama tem como objetivo ilustrar, de maneira clara e objetiva, a estrutura metodológica geral que orientou a condução da pesquisa.

Figura 1 - Metodologia geral do trabalho

Fonte: elaborado pelo autor

### 3.1.1 Definição do problema e objetivos da previsão

Este trabalho tem como ponto de partida uma necessidade prática observada em um dos produtos desenvolvidos pela empresa onde atuo, voltado à análise e visualização de dados corporativos. Especificamente, trata-se de um dashboard construído na ferramenta Power BI, que apresenta diversas análises de desempenho, incluindo uma medida responsável por estimar o faturamento do mês corrente com base nos dados registrados desde o primeiro dia do mês até o momento da consulta.

O problema que este trabalho propõe a investigar consiste em avaliar se é possível aprimorar essa estimativa por meio da aplicação de modelos de aprendizado de máquina e métodos estatísticos avançados. Para isso, foram desenvolvidos diferentes modelos preditivos (ARIMA, Theta, Suavização Exponencial e XGBoost) utilizando os mesmos dados disponíveis no dashboard, buscando simular o contexto real de previsão. O desempenho de cada modelo foi avaliado com base em métricas estatísticas padronizadas.

O objetivo principal deste estudo é verificar qual dos modelos testados apresenta melhor desempenho preditivo. A adoção do melhor modelo poderá resultar em previsões mais precisas e na geração de insights mais robustos e estratégicos.

### 3.1.2 Coleta e pré-processamento dos dados

A coleta e o pré-processamento dos dados utilizados neste trabalho foram realizadas através da ferramenta Visual Studio Code. Os dados empregados correspondem às séries históricas de faturamento disponíveis em um produto interno da empresa, sendo originalmente utilizados em um dashboard desenvolvido em Power BI.

Os dados utilizados neste estudo consistiram em registros transacionais de vendas contendo 37.425 transações no período de 2014 a 2025. Os campos principais incluíram a data de emissão do pedido, valor líquido da venda, identificação do cliente e tipo de operação comercial.

#### 3.1.2.1 Filtragem e agregação inicial

O processo de pré-processamento iniciou com a filtragem exclusiva de transações classificadas como "VENDA", excluindo devoluções e outros tipos de operações comerciais. O valor líquido das vendas foi estabelecido como variável target, representando a quantidade que os modelos tentariam prever.

#### 3.1.2.2 Anonimização dos dados

Para garantir a privacidade e conformidade com requisitos de proteção de dados, foi implementado um processo de anonimização utilizando função de hash criptográfico MD5 para transformar identificações de clientes em códigos anônimos. O sistema gerou identificadores no formato "CLIENTE_####" onde os quatro dígitos foram derivados deterministicamente do hash do nome original. Esta abordagem protegeu a privacidade dos clientes enquanto preservou a capacidade de rastreamento consistente ao longo do tempo.

#### 3.1.2.3 Agregação temporal mensal

Após a filtragem inicial, os dados transacionais foram agregados temporalmente em períodos mensais, calculando a soma total de vendas para cada mês. Este processo foi fundamental pois os modelos de séries temporais operam com observações sequenciais regularmente espaçadas no tempo.

O procedimento consistiu em agrupar todas as transações por mês e ano, gerando uma série temporal com frequência mensal cobrindo o período completo dos dados. Cada observação representou o faturamento total do mês correspondente, resultando em aproximadamente 132 pontos temporais mensais.

#### 3.1.2.4 Conversão para formato Darts

Os dados agregados foram então convertidos para o formato TimeSeries da biblioteca Darts, utilizada para implementação de todos os modelos neste estudo. A biblioteca Darts oferece uma interface unificada para modelagem de séries temporais, suportando tanto métodos estatísticos tradicionais (ARIMA, Theta, Suavização Exponencial) quanto algoritmos de machine learning (XGBoost) especializados em séries temporais.

Esta conversão incluiu a definição adequada do índice temporal (datas mensais no formato ISO), especificação da coluna de valores (faturamento mensal agregado), e configuração da frequência da série temporal (mensal). A estrutura TimeSeries permitiu que todos os modelos acessassem funcionalidades avançadas como divisão temporal apropriada, geração automática de features, e aplicação de transformações específicas para cada algoritmo.

#### 3.1.2.5 Considerações sobre engenharia de features

Diferentemente de abordagens tradicionais que requerem engenharia manual extensiva de features (criação de lags, médias móveis, codificações trigonométricas, etc.), a biblioteca Darts realiza automaticamente a criação das features necessárias para cada tipo de modelo durante o processo de treinamento.

Para os modelos estatísticos (ARIMA, Theta, Suavização Exponencial), a Darts opera diretamente sobre a série temporal univariada, aplicando internamente as transformações e diferenciações necessárias.

Para o modelo XGBoost, a Darts utiliza o módulo `XGBModel`, que cria automaticamente features temporais através de:
- Lags configuráveis da variável target
- Lags de covariadas passadas (quando aplicável)
- Encoders temporais (mês, ano, trimestre, dia do ano, semana do ano, dia da semana)
- Normalização apropriada via MaxAbsScaler

Esta abordagem simplificou significativamente o pipeline de pré-processamento, eliminando a necessidade de engenharia manual de features e garantindo consistência na preparação dos dados para todos os modelos.

### 3.1.3 Análise exploratória e estruturação da série temporal

A análise exploratória de dados (EDA) constitui uma etapa fundamental no processo de modelagem de séries temporais, precedendo a aplicação de modelos preditivos e fornecendo insights essenciais sobre a estrutura, padrões e características dos dados históricos. Conforme destacado por Bezerra (2006), a compreensão adequada do comportamento temporal dos dados é crucial para a seleção e parametrização apropriada de modelos de previsão, influenciando diretamente a qualidade e confiabilidade dos resultados obtidos.

No contexto de séries temporais de vendas, a EDA assume particular importância devido à complexidade inerente desses dados, que frequentemente apresentam componentes de tendência, sazonalidade, ciclos econômicos e variações irregulares. Segundo Makridakis, Wheelwright e Hyndman (1999), a identificação precisa desses componentes através de técnicas exploratórias adequadas é fundamental para orientar as decisões metodológicas subsequentes, incluindo a escolha de modelos estatísticos apropriados e a definição de estratégias de pré-processamento.

#### 3.1.3.1 Visão geral da série temporal

A análise exploratória foi implementada através de um sistema automatizado de visualizações desenvolvido em Python, utilizando bibliotecas especializadas em análise de séries temporais. Os dados utilizados correspondem à série temporal de vendas mensais no período de outubro de 2014 a setembro de 2025, totalizando 132 observações após o pré-processamento e agregação temporal mensal.

A estruturação dos dados seguiu as diretrizes estabelecidas por Parzen (1961), que define uma série temporal como um conjunto de observações dispostas cronologicamente, representada matematicamente como um processo estocástico. Para garantir a adequação dos dados à análise temporal, foi implementada uma verificação rigorosa da ordenação cronológica, tratamento de valores ausentes e validação da consistência temporal.

A primeira análise apresenta uma visão geral abrangente da série temporal, incluindo a evolução das vendas ao longo do tempo com linha de tendência, distribuição dos valores por ano através de gráficos de boxplot, análise das vendas acumuladas e volatilidade temporal. Esta visão panorâmica revelou uma tendência de crescimento consistente de 2014 a 2022, seguida por um declínio significativo entre os anos 2023 e 2025, com valores variando de aproximadamente R$ 1 milhão em 2014 para um pico acima de R$ 80 milhões em 2022.

Figura 3 - Visão geral da série temporal

Fonte: elaborado pelo autor

#### 3.1.3.2 Decomposição STL

A decomposição STL (Seasonal-Trend using Loess) foi aplicada para separar os componentes estruturais da série temporal. A decomposição confirmou a presença de uma tendência de longo prazo bem definida e padrões sazonais consistentes, com a série original mostrando crescimento exponencial até 2022, seguido por declínio acentuado. O componente sazonal revelou padrões regulares de variação mensal, enquanto o resíduo indicou períodos de maior volatilidade, especialmente durante os anos de transição econômica.

Figura 4 - Decomposição da série temporal

Fonte: elaborado pelo autor

#### 3.1.3.3 Análise de sazonalidade

A análise sazonal detalhada examinou os padrões mensais e de autocorrelação da série temporal. Foram calculadas as médias mensais históricas, revelando que determinados meses apresentam consistentemente maiores volumes de vendas. A análise de autocorrelação identificou dependências temporais significativas até o lag 12, confirmando a presença de sazonalidade anual na série.

Figura 5 - Análise da sazonalidade

Fonte: elaborado pelo autor

#### 3.1.3.4 Propriedades estatísticas

A análise das propriedades estatísticas incluiu o cálculo das funções de autocorrelação (ACF) e autocorrelação parcial (PACF), fundamentais para a parametrização de modelos ARIMA. A ACF mostrou correlações significativas nos primeiros lags, decaindo gradualmente até o lag 12, enquanto a PACF apresentou cortes abruptos após o primeiro lag, sugerindo características autorregressivas na série. A análise da série diferenciada (primeira diferença) confirmou a remoção da tendência, tornando a série mais adequada para modelagem estatística.

Figura 6 - Propriedades estatísticas da série temporal

Fonte: elaborado pelo autor

#### 3.1.3.5 Análise de distribuição

A análise de distribuição dos valores de vendas incluiu histograma com sobreposição de distribuição normal, gráfico Q-Q para teste de normalidade, box plot para identificação de outliers, e comparação de densidade. Os resultados indicaram que a distribuição das vendas não segue uma distribuição normal, apresentando assimetria positiva e presença de valores extremos.

Figura 7 - Análise de distribuição

Fonte: elaborado pelo autor

#### 3.1.3.6 Evolução temporal detalhada

A análise de evolução temporal examinou as taxas de crescimento anual, padrões sazonais por ano, e tendência linear geral. O cálculo das taxas de crescimento revelou crescimento superior a 200% em 2015, estabilização em torno de 20 a 40% nos anos intermediários, e declínios acentuados nos anos finais.

Figura 8 - Evolução temporal das vendas

Fonte: elaborado pelo autor

#### 3.1.3.7 Análise de correlação temporal

A análise de correlação incluiu correlações com lags de 1 a 12 meses, autocorrelação parcial detalhada, matriz de correlação para lags selecionados e correlação com componentes temporais (ano, trimestre, mês). Os resultados mostraram correlações elevadas (>0,8) para os primeiros lags, decaindo gradualmente até o lag 12. A matriz de correlação dos lags selecionados revelou padrões de dependência temporal que orientaram a configuração dos modelos preditivos.

Figura 9 - Análise de correlação temporal

Fonte: elaborado pelo autor

#### 3.1.3.8 Insights para modelagem

Com base nesta análise exploratória abrangente, foram identificados os seguintes resultados fundamentais para a modelagem preditiva:

a) **Estacionariedade**: A série original não é estacionária devido à forte tendência, requerendo diferenciação para modelos ARIMA (d = 1);

b) **Sazonalidade**: Presença confirmada de sazonalidade anual (período 12) com padrões consistentes;

c) **Autocorrelação**: Dependências temporais significativas até 12 lags, orientando a parametrização dos modelos;

d) **Distribuição**: Dados não seguem distribuição normal, apresentando assimetria positiva;

e) **Tendência**: Tendência de longo prazo bem definida com crescimento até 2022 seguido de declínio;

f) **Volatilidade**: Variação da volatilidade ao longo do tempo, com períodos de maior instabilidade.

Estes resultados orientaram diretamente a configuração dos parâmetros para cada modelo preditivo, a escolha das técnicas de pré-processamento específicas, e as estratégias de validação temporal adotadas nas etapas subsequentes.

## 3.2 MODELOS DE PREVISÃO UTILIZADOS

A modelagem preditiva é a etapa central deste trabalho, sendo responsável por transformar os dados estruturados em previsões quantitativas para o faturamento. Considerando as diferentes abordagens e características dos dados, foram selecionados múltiplos modelos de previsão, cada um com suas próprias vantagens, desvantagens e características específicas de implementação.

Os modelos escolhidos para este estudo incluem técnicas tradicionais de séries temporais, como ARIMA, Theta e Suavização Exponencial, bem como o algoritmo XGBoost, amplamente utilizado em aplicações empresariais para problemas de previsão com séries temporais. Cada um desses modelos foi avaliado quanto à sua capacidade de capturar padrões históricos, prever tendências futuras e lidar com os desafios típicos desse tipo de dado, como sazonalidade, tendência e variações irregulares.

Para garantir uma análise comparativa robusta, foram considerados fatores como a facilidade de implementação, complexidade computacional e a precisão das previsões geradas. Todos os modelos foram implementados utilizando a biblioteca Darts, que oferece uma interface unificada e padronizada para modelagem de séries temporais, garantindo consistência na preparação dos dados, divisão temporal e avaliação de desempenho.

Nos subtópicos a seguir, cada modelo é apresentado individualmente, incluindo os requisitos específicos de implementação e o diagrama do fluxo metodológico correspondente.

### 3.2.1 ARIMA

A figura a seguir mostra a metodologia utilizada para o modelo.

Figura 10 - Metodologia do modelo ARIMA

Fonte: elaborado pelo autor

#### 3.2.1.1 Importação das bibliotecas e configuração do ambiente

A implementação do modelo ARIMA foi realizada utilizando o Visual Studio Code como ambiente de desenvolvimento integrado, garantindo controle de versão e reprodutibilidade do código. O ambiente Python foi configurado com as seguintes bibliotecas essenciais:

a) **Darts**: Biblioteca especializada em séries temporais que forneceu o módulo ARIMA (com seleção automática de parâmetros via AutoARIMA), métodos de divisão temporal apropriados para séries temporais, e funções integradas de avaliação e diagnóstico.

b) **Pandas**: Utilizado para manipulação e estruturação inicial dos dados, conversão de tipos de dados temporais, e operações de agregação e filtragem durante o pré-processamento.

c) **Matplotlib e Seaborn**: Empregados para geração de visualizações diagnósticas, incluindo gráficos de série temporal, correlogramas, análise de resíduos e comparações entre valores observados e previstos.

Esta preparação foi fundamental para garantir que todas as operações subsequentes fossem executadas de forma padronizada e rastreável.

#### 3.2.1.2 Ingestão e conversão dos dados para série temporal

O processo de ingestão iniciou com o carregamento dos dados de faturamento mensal previamente processados na etapa 3.1.2, obtidos do arquivo CSV estruturado com 132 observações mensais. Os dados foram validados quanto à:

a) **Integridade temporal**: Verificação de continuidade mensal sem lacunas, confirmação da ordenação cronológica correta, e validação do formato de datas no padrão ISO (YYYY-MM-DD).

b) **Qualidade dos valores**: Identificação de valores nulos, negativos ou extremos que poderiam comprometer a modelagem, e confirmação da escala monetária consistente (valores em reais).

c) **Estrutura adequada**: Configuração do índice temporal como DatetimeIndex do Pandas, garantindo operações temporais apropriadas.

A conversão para o objeto TimeSeries da Darts foi realizada especificando a coluna de valores (faturamento mensal), o índice temporal (datas mensais), e a frequência da série ('MS' para mensal). Esta estrutura otimizada permitiu que o modelo ARIMA acessasse funcionalidades avançadas como detecção automática de periodicidade sazonal, aplicação de transformações temporais (diferenciação), e geração de previsões de forma eficiente.

#### 3.2.1.3 Verificação de estacionaridade e diferenciação

A avaliação de estacionariedade foi conduzida considerando os achados da análise exploratória que evidenciaram forte tendência não linear (crescimento exponencial até 2022, seguido de declínio acentuado) e padrões sazonais anuais consistentes.

a) **Testes de estacionariedade**: O AutoARIMA da Darts realiza testes internos (ADF - Augmented Dickey-Fuller) para detectar a presença de raiz unitária e determinar automaticamente a necessidade de diferenciação.

b) **Estratégia de diferenciação**: O AutoARIMA foi configurado para explorar automaticamente:

   - **Diferenciação não sazonal (d)**: Testadas ordens de 0 a 2, sendo d = 1 (primeira diferença) a mais comum para remover tendência linear.

   - **Diferenciação sazonal (D)**: Avaliada com período 12 (sazonalidade anual), testando D = 0 (sem diferenciação sazonal) e D = 1 (uma diferenciação sazonal para remover padrões sazonais não estacionários).

O processo de diferenciação foi crucial para transformar a série não estacionária original em uma série com propriedades estatísticas estáveis, evitando regressões espúrias e garantindo a validade dos pressupostos do modelo ARIMA. A biblioteca Darts aplicou estas transformações de forma automática e reversível para as previsões finais.

#### 3.2.1.4 Divisão dos dados em conjuntos de treino e teste

A divisão temporal foi implementada seguindo rigorosamente o princípio de não-sobreposição temporal, essencial para validação realística de modelos de séries temporais. A estratégia adotada foi:

a) **Conjunto de treino**: Primeiros 80% da série (aproximadamente 105 meses), representando o período de outubro de 2014 até meados de 2023. Este período incluiu a fase de crescimento consistente e o pico histórico das vendas, fornecendo ao modelo informação suficiente sobre tendências de longo prazo e padrões sazonais estabelecidos.

b) **Conjunto de teste**: Últimos 20% da série (aproximadamente 27 meses), correspondendo ao período final até setembro de 2025. Este período capturou a fase de declínio das vendas, representando um desafio real de generalização para o modelo.

c) **Justificativa da divisão**: A proporção 80/20 foi escolhida para garantir quantidade suficiente de dados para o treinamento (especialmente importante para capturar múltiplos ciclos sazonais anuais), ao mesmo tempo que preservou um horizonte de teste representativo para avaliar performance preditiva.

A implementação utilizou métodos nativos da Darts, que garantiram preservação da estrutura temporal e evitaram vazamento de informações futuras para o conjunto de treino.

#### 3.2.1.5 Definição dos parâmetros p, d e q

A parametrização do modelo foi conduzida através do AutoARIMA da Darts, que implementou uma busca sistemática e otimizada pelos melhores parâmetros SARIMA(p,d,q)(P,D,Q)s. Os parâmetros foram definidos como:

a) **Parâmetros não sazonais**:

   - **p (ordem autorregressiva)**: Número de lags da série defasada utilizados como preditores. Testadas ordens de 0 a 5, onde p = 1 indica dependência do valor anterior, p = 2 inclui os dois valores anteriores, etc.

   - **d (ordem de diferenciação)**: Número de diferenciações aplicadas para tornar a série estacionária. Avaliadas ordens de 0 a 2, baseadas nos testes de estacionariedade.

   - **q (ordem de média móvel)**: Número de erros de previsão defasados incluídos no modelo. Testadas ordens de 0 a 5, capturando dependências nos termos de erro.

b) **Parâmetros sazonais (período s = 12)**:

   - **P (autorregressivo sazonal)**: Dependência de valores sazonais defasados (ex.: mesmo mês do ano anterior). Testadas ordens de 0 a 2.

   - **D (diferenciação sazonal)**: Diferenciação aplicada com período sazonal para remover não estacionariedade sazonal. Avaliadas ordens de 0 a 1.

   - **Q (média móvel sazonal)**: Erros sazonais defasados incluídos no modelo. Testadas ordens de 0 a 2.

Para critério de seleção, o AutoARIMA utilizou o AIC (Akaike Information Criterion) para balancear qualidade do ajuste com parcimônia do modelo, selecionando automaticamente a configuração que minimizou o AIC. O algoritmo implementou busca stepwise para eficiência computacional, explorando configurações vizinhas de forma inteligente.

#### 3.2.1.6 Treinamento do modelo

O processo de treinamento foi executado após a seleção automática dos melhores parâmetros, utilizando os algoritmos de estimação implementados na Darts. O treinamento envolveu:

a) **Estimação por máxima verossimilhança**: Os coeficientes do modelo foram estimados através da maximização da função de verossimilhança, que encontrou os parâmetros que melhor explicaram os dados observados no conjunto de treino.

b) **Otimização numérica**: O processo utilizou algoritmos de otimização não linear para encontrar os valores ótimos dos coeficientes, iniciando de valores iniciais estimados e iterando até convergência.

c) **Ajuste da componente sazonal**: O modelo SARIMA ajustou simultaneamente os padrões não sazonais (tendência de curto prazo, dependências de lags próximos) e sazonais (padrões anuais, dependências de períodos equivalentes em anos anteriores).

d) **Validação do ajuste**: Durante o treinamento, foram monitoradas métricas de convergência e estabilidade dos coeficientes estimados para garantir adequação do processo de otimização.

O resultado foi um modelo completamente parametrizado, capaz de capturar tanto as dependências temporais de curto prazo quanto os padrões sazonais anuais identificados na análise exploratória.

#### 3.2.1.7 Validação do modelo e ajustes finos

A etapa de validação consistiu na geração de previsões para todo o horizonte do conjunto de teste e avaliação sistemática da performance preditiva:

a) **Geração de previsões**: O modelo treinado foi utilizado para produzir previsões recursivas, onde cada previsão utilizou apenas informações disponíveis até aquele ponto temporal. Este processo simulou fielmente o cenário real de previsão operacional.

b) **Intervalos de confiança**: Foram gerados intervalos de previsão (tipicamente 95% de confiança) baseados na variância estimada dos erros do modelo, fornecendo medida de incerteza associada a cada previsão.

c) **Métricas de avaliação**: A performance foi avaliada através do conjunto padronizado de métricas:

   - **MAE (Mean Absolute Error)**: Erro absoluto médio em reais, interpretável diretamente na escala do problema.

   - **RMSE (Root Mean Squared Error)**: Raiz do erro quadrático médio, penalizando mais fortemente grandes desvios.

   - **MAPE (Mean Absolute Percentage Error)**: Erro percentual absoluto médio, permitindo interpretação relativa independente da escala.

d) **Análise temporal das previsões**: Foi conduzida análise período a período para identificar padrões nos erros, sazonalidade residual, e performance diferencial ao longo do horizonte de previsão.

#### 3.2.1.8 Análise residual

Uma análise detalhada dos resíduos do modelo foi conduzida para verificar se os erros de previsão se distribuíram de forma aleatória, sem padrões sistemáticos não modelados. Foram gerados gráficos de autocorrelação (ACF) e autocorrelação parcial (PACF) dos resíduos, buscando confirmar comportamento próximo ao ruído branco.

Resíduos com padrões significativos indicaram que o modelo não conseguiu capturar completamente as relações temporais nos dados. Adicionalmente, a análise incluiu inspeção visual da distribuição dos resíduos e identificação de outliers ou eventos atípicos que poderiam comprometer a precisão das previsões futuras. Esta validação foi essencial para confirmar a adequação do modelo selecionado.

#### 3.2.1.9 Armazenamento dos resultados para comparação futura

Foram geradas visualizações específicas para documentar o desempenho do modelo ARIMA, incluindo gráficos de série temporal comparando valores observados e previstos, análise de resíduos ao longo do tempo e representação gráfica da estrutura de correlação do conjunto de dados para diagnóstico.

Os resultados do modelo ARIMA, incluindo previsões, métricas de desempenho, parâmetros selecionados e diagnósticos, foram salvos de forma estruturada para posterior comparação com os demais modelos (Theta, Suavização Exponencial e XGBoost). Esta documentação foi essencial para a análise comparativa final e escolha da abordagem preditiva mais adequada.

### 3.2.2 Suavização Exponencial

A figura a seguir mostra a metodologia utilizada para o modelo.

Figura 11 - Metodologia do modelo Suavização Exponencial

Fonte: elaborado pelo autor

O modelo de Suavização Exponencial compartilhou grande parte da metodologia com o ARIMA, diferindo principalmente na abordagem de modelagem e nos critérios de seleção do modelo. As etapas de importação de bibliotecas, ingestão e conversão de dados e a divisão treino/teste foram executadas de forma idêntica ao ARIMA, utilizando a mesma biblioteca Darts, mesma estrutura TimeSeries, e mesma proporção 80/20 com divisão temporal rigorosa.

#### 3.2.2.1 Análise de componentes para seleção do modelo

Diferentemente do ARIMA, que se baseou em testes de estacionariedade e análise de correlogramas, o modelo de Suavização Exponencial utilizou os resultados da decomposição STL já realizada na análise exploratória para orientar a seleção do tipo apropriado de modelo.

Com base nos componentes já extraídos na EDA, a biblioteca Darts implementou critérios automáticos para escolha entre:

a) **Suavização Exponencial Simples (SES)**: Para séries sem tendência ou sazonalidade significativas.

b) **Método de Holt**: Para séries com tendência forte, mas sazonalidade fraca.

c) **Método de Holt-Winters**: Para séries com ambos os componentes significativos (caso esperado desta série).

#### 3.2.2.2 Decisão entre modelo aditivo e multiplicativo

Uma etapa específica da Suavização Exponencial foi a escolha entre formulações aditiva e multiplicativa, baseada na análise dos componentes sazonais da EDA:

a) **Modelo Aditivo**: Selecionado quando a amplitude da sazonalidade permaneceu relativamente constante ao longo do tempo.

b) **Modelo Multiplicativo**: Selecionado quando a amplitude da sazonalidade variou proporcionalmente ao nível da série.

A decisão foi automatizada pela Darts baseada na análise da variância relativa dos componentes sazonais já extraídos na EDA.

#### 3.2.2.3 Configuração e otimização de parâmetros

Ao contrário do ARIMA, que utilizou parâmetros discretos (p, d, q), a Suavização Exponencial otimizou parâmetros contínuos de suavização:

a) **Parâmetros do modelo Holt-Winters**:
   - **α (alfa)**: Parâmetro de suavização do nível (0 < α ≤ 1)
   - **β (beta)**: Parâmetro de suavização da tendência (0 ≤ β ≤ 1)
   - **γ (gama)**: Parâmetro de suavização sazonal (0 ≤ γ ≤ 1)

b) **Período sazonal**: Fixado em 12 meses conforme evidenciado na EDA.

c) **Processo de otimização**: A Darts utilizou algoritmos de minimização numérica para encontrar os valores ótimos que minimizaram o erro quadrático médio no conjunto de treino.

#### 3.2.2.4 Treinamento por suavização recursiva

O processo de treinamento diferiu fundamentalmente do ARIMA por utilizar suavização exponencial recursiva ao invés de estimação de máxima verossimilhança:

a) **Inicialização dos componentes**:
   - Nível inicial estimado como média dos primeiros períodos
   - Tendência inicial calculada como diferença média inicial
   - Índices sazonais estimados através dos primeiros ciclos da série

b) **Atualização recursiva**: Para cada período t do treino, os componentes foram atualizados através de combinações ponderadas dos valores observados e componentes anteriores projetados.

Este processo iterativo permitiu ao modelo adaptar-se gradualmente aos padrões, diferindo da estimação simultânea de todos os parâmetros no ARIMA.

#### 3.2.2.5 Geração de previsões diretas

A geração de previsões na Suavização Exponencial utilizou abordagem direta (não recursiva) baseada nos componentes finais, projetando o nível futuro adicionando tendência multiplicada pelo horizonte ao último nível, e obtendo o componente sazonal do índice correspondente ao período do ano.

#### 3.2.2.6 Análise residual específica para suavização

A análise residual seguiu protocolo similar ao ARIMA, mas com focos específicos na validação de componentes (tendência e sazonalidade), estabilidade dos parâmetros otimizados (α, β e γ), e adequação do modelo selecionado (aditivo vs. multiplicativo) através de análise visual dos resíduos padronizados e métricas de ajuste.

### 3.2.3 Theta

O modelo Theta compartilhou as etapas fundamentais de preparação com os modelos anteriores, diferindo principalmente na abordagem de decomposição e extrapolação. As etapas de importação de bibliotecas, ingestão e conversão de dados e divisão treino/teste foram executadas de forma idêntica aos modelos anteriores, utilizando a mesma biblioteca Darts, mesma estrutura TimeSeries, e mesma divisão temporal 80/20.

A figura a seguir mostra a metodologia utilizada para o modelo.

Figura 12 - Metodologia do modelo Theta

Fonte: elaborado pelo autor

#### 3.2.3.1 Verificação de pré-condições do método Theta

O método Theta na biblioteca Darts exigiu verificações específicas antes da aplicação:

a) **Validação da série temporal**: Confirmação da ausência de valores nulos na série, pois o Theta da Darts não possui tratamento automático para dados ausentes.

b) **Verificação de univariância**: O método foi aplicado exclusivamente à série temporal univariada de faturamento mensal, sem variáveis explicativas adicionais, seguindo a natureza original do método proposto por Assimakopoulos e Nikolopoulos (2000).

c) **Confirmação de regularidade temporal**: Verificação da frequência mensal constante da série, requisito para a decomposição Theta funcionar adequadamente.

#### 3.2.3.2 Configuração automática do modelo

O método Theta da Darts ofereceu configuração totalmente automática:

a) **Parâmetro Theta (θ)**: A Darts implementou seleção automática do parâmetro θ, que controla a curvatura das linhas Theta. Valores θ < 1 enfatizam tendências de longo prazo, enquanto θ > 1 destacam variações de curto prazo.

b) **Detecção automática de sazonalidade**: O Theta detectou automaticamente a presença e o período da sazonalidade (12 meses) com base nos padrões da série.

c) **Configuração de decomposição**: O modelo foi configurado para aplicar decomposição automática da série em componentes Theta, sem necessidade de especificação manual.

#### 3.2.3.3 Decomposição e criação das linhas Theta

Esta etapa foi específica do método Theta:

a) **Aplicação das segundas diferenças**: O método aplicou o operador de segundas diferenças à série original conforme a formulação matemática de Assimakopoulos e Nikolopoulos (2000).

b) **Geração das linhas Theta**: Foram criadas múltiplas linhas Theta através de transformações matemáticas, incluindo:
   - Linha Theta 0 (θ = 0): Representa tendência linear de longo prazo
   - Linha Theta 2 (θ = 2): Captura variações de curto prazo e sazonalidade

#### 3.2.3.4 Treinamento e ajuste das componentes

O processo de treinamento do Theta diferiu dos outros modelos:

a) **Ajuste das linhas individuais**: Cada linha Theta foi ajustada separadamente:
   - Linha Theta 0: Ajustada por regressão linear para capturar tendência de longo prazo
   - Linha Theta 2: Ajustada por Suavização Exponencial Simples (SES) para variações de curto prazo

b) **Otimização automática**: A Darts implementou otimização automática dos parâmetros de cada componente.

#### 3.2.3.5 Combinação de previsões e extrapolação

A geração de previsões seguiu abordagem única de combinação de extrapolações, onde cada linha Theta foi extrapolada separadamente para o horizonte de teste, e as previsões finais foram obtidas através de combinação ponderada das extrapolações individuais, tipicamente com pesos iguais ou otimizados baseados na performance histórica.

#### 3.2.3.6 Avaliação e diagnósticos específicos

A avaliação seguiu protocolo similar aos modelos anteriores, com análises específicas de validação das linhas Theta, verificação da capacidade de reconstrução da série original, e análise de estabilidade dos parâmetros otimizados.

### 3.2.4 XGBoost

A figura a seguir mostra a metodologia utilizada para o modelo.

Figura 13 - Metodologia do modelo XGBoost

Fonte: elaborado pelo autor

#### 3.2.4.1 Preparação e integração com Darts

O modelo XGBoost foi implementado utilizando o módulo `XGBModel` da biblioteca Darts, que integra o algoritmo XGBoost com a infraestrutura de séries temporais da Darts. Diferentemente da implementação tradicional que requer engenharia manual extensiva de features, o `XGBModel` da Darts automatiza a criação de features temporais necessárias para o treinamento.

A entrada do modelo foi a mesma série temporal univariada utilizada pelos outros modelos (faturamento mensal agregado), mantendo consistência na preparação dos dados. A Darts se encarregou automaticamente de transformar esta série temporal em formato tabular apropriado para o XGBoost durante o processo de treinamento.

#### 3.2.4.2 Divisão dos dados em treino e teste

Assim como nos demais modelos, os dados foram divididos respeitando rigorosamente a ordem cronológica na proporção 80/20, evitando vazamento de informações futuras. A Darts garantiu que a divisão temporal fosse consistente com os outros modelos implementados.

#### 3.2.4.3 Engenharia automática de features

O `XGBModel` da Darts criou automaticamente as features necessárias através de parâmetros configuráveis:

a) **Lags da variável target**: Foram configurados 17 lags principais [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -15, -18, -24, -30, -36] para capturar dependências temporais em diferentes horizontes.

b) **Lags de covariadas passadas**: Configurados 8 lags [-1, -2, -3, -4, -5, -6, -12, -24] para capturar padrões adicionais de dependência temporal.

c) **Encoders temporais**: Foram adicionados automaticamente 6 encoders temporais (month, year, quarter, dayofyear, weekofyear, dayofweek) para capturar padrões cíclicos e sazonais.

d) **Normalização**: Aplicada automaticamente via MaxAbsScaler para garantir escala apropriada das features, particularmente importante para lidar com outliers em dados de vendas.

Esta abordagem eliminou a necessidade de criar manualmente features como médias móveis, codificações trigonométricas, e estatísticas agregadas, simplificando significativamente o pipeline e garantindo que apenas as features mais relevantes fossem utilizadas.

#### 3.2.4.4 Configuracao dos hiperparametros

O modelo XGBoost implementado via Darts separou os parametros em duas categorias distintas: parametros especificos do framework Darts para processamento de series temporais e hiperparametros do algoritmo XGBoost propriamente dito.

**Parametros do framework Darts (configuracao de series temporais):**

a) **lags**: 17 valores de defasagem [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -15, -18, -24, -30, -36] para capturar dependencias temporais em multiplos horizontes.

b) **lags_past_covariates**: 8 lags adicionais [-1, -2, -3, -4, -5, -6, -12, -24] para padroes de dependencia temporal complementares.

c) **add_encoders**: Encoders temporais automaticos incluindo month, year, quarter, dayofyear, weekofyear e dayofweek para captura de padroes ciclicos e sazonais.

d) **data_scaling**: MaxAbsScaler aplicado automaticamente para normalizacao robusta das features.

**Hiperparametros do algoritmo XGBoost (passados via kwargs):**

e) **n_estimators**: 2000 arvores de decisao para garantir capacidade adequada de aprendizado e convergencia do algoritmo de gradient boosting.

f) **max_depth**: 8 niveis de profundidade maxima, controlando a complexidade das arvores individuais e evitando overfitting.

g) **learning_rate**: 0.05 para controlar o peso de cada nova arvore no ensemble, garantindo aprendizado estavel e convergencia gradual.

h) **subsample**: 0.9 (90% de amostragem) para aumentar a generalizacao do modelo atraves de variacao estocastica nas amostras de treinamento.

i) **colsample_bytree**: 0.9 para selecionar aleatoriamente 90% das features em cada arvore, promovendo diversidade no ensemble.

j) **reg_alpha**: 0.2 (regularizacao L1/Lasso) para penalizar complexidade e promover esparsidade nos pesos do modelo.

k) **reg_lambda**: 1.5 (regularizacao L2/Ridge) para controle adicional de complexidade e suavizacao dos pesos.

l) **random_state**: 42 para garantir reprodutibilidade total dos resultados entre execucoes.

Esta configuracao hibrida aproveitou a especializacao da Darts em processamento de series temporais (geracao automatica de lags e encoders temporais) combinada com o poder preditivo do algoritmo XGBoost (ensemble de arvores com gradient boosting). Os hiperparametros do XGBoost foram definidos manualmente com base em praticas estabelecidas para modelos de previsao, priorizando capacidade de aprendizado (n_estimators alto e max_depth moderado) equilibrada com regularizacao (reg_alpha e reg_lambda) para evitar overfitting.

#### 3.2.4.5 Treinamento do modelo

O processo de treinamento do XGBoost seguiu o paradigma de gradient boosting:

a) **Inicialização**: O processo iniciou com uma previsão inicial simples (geralmente a média dos valores de treino).

b) **Treinamento iterativo**: Em cada iteração, uma nova árvore de decisão foi treinada para modelar os resíduos (erros) das árvores anteriores, corrigindo gradualmente as falhas do modelo.

c) **Atualização das previsões**: As previsões foram atualizadas somando as previsões das novas árvores às previsões acumuladas das árvores anteriores, multiplicadas pela taxa de aprendizado (learning_rate).

d) **Regularização**: Durante o treinamento, os termos de regularização L1 e L2 foram aplicados para penalizar complexidade excessiva e promover modelos mais simples e generalizáveis.

A integração com Darts garantiu que todo este processo respeitasse a natureza temporal dos dados, utilizando apenas informações disponíveis até cada ponto temporal durante o treinamento.

#### 3.2.4.6 Avaliação inicial de desempenho

A avaliação do desempenho foi realizada de maneira análoga aos outros modelos, através das métricas MAE, RMSE e MAPE aplicadas ao conjunto de teste. A análise dos erros permitiu verificar a capacidade do modelo em capturar padrões complexos presentes nos dados de vendas.

#### 3.2.4.7 Validação e análise de resultados

Foi empregada validação temporal adequada a séries temporais, assegurando a robustez dos resultados e a ausência de overfitting. Os resultados da validação foram analisados quanto à consistência e possíveis padrões residuais, confirmando a adequação do modelo.

#### 3.2.4.8 Geração das previsões finais e armazenamento dos resultados

As previsões finais geradas pelo modelo XGBoost foram armazenadas em formato estruturado para comparação direta com os resultados dos demais modelos (ARIMA, Theta e Suavização Exponencial), permitindo análise comparativa abrangente baseada nas mesmas métricas padronizadas.

## 3.3 AVALIAÇÃO E COMPARAÇÃO DOS MODELOS

Após o ajuste e validação de todos os modelos preditivos considerados neste trabalho, foi realizada uma comparação quantitativa do desempenho de cada modelo utilizando as seguintes métricas estatísticas, recomendadas pela literatura para problemas de previsão de séries temporais:

a) **Erro Médio Absoluto (MAE)**: Mede o erro médio absoluto entre os valores previstos e os valores reais, sendo interpretável diretamente na escala monetária do problema.

b) **Raiz do Erro Quadrático Médio (RMSE)**: Penaliza mais fortemente erros grandes, sendo útil quando desvios significativos são críticos para o negócio.

c) **Erro Percentual Absoluto Médio (MAPE)**: Expressa o erro em termos percentuais, facilitando a interpretação relativa do desempenho independente da escala dos dados.

Essas métricas foram calculadas para o conjunto de teste de cada modelo, permitindo comparação justa entre as diferentes abordagens. O modelo que apresentou o menor valor de erro (considerando principalmente MAE e RMSE) foi selecionado como o modelo de melhor desempenho, conforme abordagem utilizada por Hyndman et al. (1999) e Gardner (1985).

A escolha final do modelo foi baseada não apenas no desempenho quantitativo, mas também na sua viabilidade de implementação e capacidade de generalização para diferentes períodos temporais, conforme recomendam Gardner (1985) e Hyndman et al. (1999).
