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

## 3.1.2 Coleta e pré-processamento dos dados

A coleta e a o pré-processamento dos dados utilizados neste trabalho foram realizadas através da ferramenta Visual Studio Code. Os dados empregados correspondem às séries históricas de faturamento disponíveis em um produto interno da empresa, sendo originalmente utilizados em um dashboard desenvolvido em Power BI.
Inicialmente, foi realizada a extração dos dados diretamente do banco de dados no formato .csv sem nenhuma transformação, assegurando a preservação de todas as informações relevantes.
O segundo passo após a coleta dos dados foi o pré-processamento, cujo objetivo é preparar as informações para a aplicação dos modelos de previsão. Esta etapa compreendeu tanto procedimentos voltados à integridade e à estrutura analítica dos dados quanto à conformidade ética e à proteção da privacidade.
No pré-processamento foi realizada, como primeira medida, a anonimização dos dados utilizando scripts desenvolvidos em Python. O processo de anonimização foi executado através de funções de hash que geram identificadores consistentes e irreversíveis para os dados dos nomes das empresas, lojas e vendedores. Os nomes das empresas foram substituídos por códigos no formato "EMPRESA_XXXX", os nomes das lojas por "LOJA_XXXX" e os nomes dos vendedores por "VENDEDOR_XXXX", onde XXXX representa um número de quatro dígitos derivado do hash original.


## 3.1.3 Pré-processamento e transformações dos dados

Após a coleta dos dados, será iniciado o processo de pré-processamento com o objetivo de preparar as informações para a aplicação dos modelos de previsão. Esta etapa  compreenderá  tanto  procedimentos  voltados  à  integridade  e  à  estrutura analítica dos dados quanto à conformidade ética e à proteção da privacidade.

Como primeira medida, será realizada a anonimização dos dados diretamente na ferramenta Visual Studio Code, utilizando scripts desenvolvidos em Python. Essa anonimização terá como finalidade remover ou substituir identificadores sensíveis por pseudônimos ou códigos aleatórios, assegurando que nenhuma informação de caráter pessoal ou sigiloso pudesse ser associada a entidades reais. Essa prática está em conformidade com os princípios de segurança de dados e com a responsabilidade profissional no manuseio de informações empresariais.

Concluída a anonimização, serão conduzidas as etapas de transformação dos dados.  Os  registros  serão  organizados  em  formato  de  série  temporal  uni  variada, respeitando a ordenação cronológica e a granularidade mensal original dos dados de faturamento. Será realizado o tratamento de valores ausentes por meio de técnicas apropriadas, como interpolação linear ou imputação por média móvel, a depender da distribuição local dos dados.

Outras etapas incluirão a padronização dos tipos de dados, normalização de nomes de colunas, e a identificação e tratamento de valores atípicos (outliers), que poderiam interferir na qualidade das previsões. Ao final desse processo, será gerado um  conjunto  de  dados  estruturado  e  limpo,  que  servirá  como  base  geral  para  a construção das versões específicas adaptadas a cada modelo preditivo.

## 3.1.5 Implementação do pipeline de processamento de dados

O processo de pré-processamento foi implementado através de um pipeline automatizado desenvolvido em Python, que executa de forma sequencial e sistemática todas as etapas de transformação dos dados. Este pipeline foi estruturado para garantir a reprodutibilidade e a consistência no tratamento dos dados em todas as execuções.

Durante o carregamento dos três arquivos CSV, foi implementada a remoção imediata da coluna CPF do conjunto de dados de vendedores, eliminando completamente esta informação sensível do pipeline de processamento. Esta abordagem garante que nenhuma informação pessoal identificável seja processada ou armazenada durante as etapas subsequentes.

O processo de anonimização foi executado através de funções de hash que geram identificadores consistentes e irreversíveis para empresas, lojas e vendedores. Os nomes das empresas foram substituídos por códigos no formato "EMPRESA_XXXX", os nomes das lojas por "LOJA_XXXX" e os nomes dos vendedores por "VENDEDOR_XXXX", onde XXXX representa um número de quatro dígitos derivado do hash original. Esta metodologia preserva a integridade referencial dos dados enquanto garante o anonimato completo das entidades envolvidas.

A consolidação dos dados foi realizada através de operações de junção entre as três tabelas, utilizando as chaves primárias apropriadas para manter a integridade relacional. Primeiro, os dados de vendas foram unidos com as informações das lojas através do código da loja, seguido pela junção com os dados dos vendedores através do código do vendedor. Estas operações foram executadas preservando todos os registros de vendas, mesmo aqueles sem correspondência nas tabelas dimensionais.

Durante o processo de consolidação, foi criada uma coluna derivada "Vendas" que representa a soma das comissões regulares e recorrentes, fornecendo uma métrica consolidada de desempenho de vendas. Foram aplicados filtros de qualidade de dados, removendo registros com datas inválidas e vendas com valores zero ou negativos, garantindo que apenas dados válidos e significativos sejam incluídos no conjunto final.

O resultado final é exportado em formato Parquet com compressão, proporcionando armazenamento eficiente e carregamento rápido para as etapas subsequentes de análise. O arquivo resultante mantém toda a estrutura e integridade dos dados originais, porém com as devidas anonimizações aplicadas.

## 3.1.6 Análise exploratória e caracterização dos dados para machine learning

Após a consolidação dos dados, foi conduzida uma análise exploratória detalhada para identificar as características estruturais e estatísticas do conjunto de dados que orientariam as estratégias específicas de pré-processamento para machine learning. Esta análise revelou um dataset com 235.735 registros e 25 colunas, totalizando aproximadamente 73,58 MB de dados.

A análise temporal demonstrou que o dataset abrange o período de janeiro de 2023 a abril de 2025, representando 849 dias de dados históricos. A distribuição temporal mostrou-se heterogênea, com 57.979 registros em 2023, 127.648 registros em 2024, e 50.108 registros no período parcial de 2025, indicando variações sazonais significativas que necessitaram tratamento específico.

A variável target, definida como a soma das colunas Comissao e ComissaoR, apresentou características estatísticas importantes: média de R$ 12.402,23, com valores variando de R$ 2,00 a R$ 110.784,00. A análise de correlação identificou forte correlação (1,0) entre a variável target e a coluna Vendas, além de correlações moderadas com Assinatura (0,70) e Comissao (0,71), indicando variáveis preditivas relevantes.

Foram identificados problemas significativos de qualidade dos dados que exigiram tratamento específico: 2.461 registros duplicados (1,04%), 200.392 valores ausentes na coluna Situacao (85,01%), 92.150 valores faltantes em Descricao (39,09%), e 6.796 valores nulos em Bloqueado_Vendedor (2,88%). Adicionalmente, detectaram-se valores zerados em grande quantidade: 176.317 zeros na coluna Valor (74,79%), 35.465 na Comissao (15,04%), e 200.270 na ComissaoR (84,96%).

A análise de outliers utilizando o método IQR identificou 2.657 valores atípicos (1,13% dos dados) na variável target, concentrados principalmente nos percentis superiores, com o percentil 99 apresentando valores de R$ 41.574,00. Essa concentração de outliers nos valores altos sugeriu a necessidade de técnicas de winsorização para evitar distorções nos modelos de machine learning.

## 3.1.7 Pipeline de pré-processamento especializado para machine learning

Com base na análise exploratória, foi desenvolvido um pipeline de pré-processamento específico para machine learning, implementado através da classe MLDataPreprocessor. Este pipeline executa dez etapas sequenciais de transformação dos dados, otimizadas para maximizar o desempenho dos algoritmos de aprendizado de máquina.

A primeira etapa consiste na criação da variável target através da soma das colunas Comissao e ComissaoR, estabelecendo a métrica principal para os problemas de regressão. A segunda etapa implementa a criação de features temporais derivadas da coluna Data, incluindo componentes lineares (ano, mês, dia, dia da semana, trimestre, dia do ano) e componentes cíclicos senoidais e cossenoidais para capturar padrões sazonais de forma contínua.

O tratamento de valores faltantes constitui a terceira etapa, aplicando estratégias específicas por tipo de variável: preenchimento de variáveis categóricas com valores padrão apropriados (Situacao com "Desconhecido", Descricao com "Sem_Descricao") e imputação de variáveis numéricas com zero para Bloqueado_Vendedor, indicando ausência de bloqueio.

A quarta etapa remove registros duplicados preservando a primeira ocorrência, garantindo a unicidade dos dados. A quinta etapa implementa o tratamento de outliers através de winsorização usando o método IQR, substituindo valores extremos pelos limites calculados para preservar a distribuição geral dos dados sem perder informações.

A sexta etapa cria features agregadas para melhorar o poder preditivo dos modelos, calculando estatísticas por vendedor, loja e período temporal (médias, desvios padrão, contagens e somatórias). Essas features agregadas capturam padrões comportamentais e tendências históricas relevantes para previsão.

A sétima etapa codifica variáveis categóricas utilizando estratégias diferenciadas: One-Hot Encoding para variáveis de baixa cardinalidade (Ativo_Loja, Tipo, CanalVendas, Situacao) e Label Encoding para variáveis de alta cardinalidade (Nome_Loja, Nome_Vendedor, Plano), otimizando a representação para diferentes tipos de algoritmos.

A oitava etapa remove colunas redundantes ou irrelevantes para machine learning, incluindo a coluna Data original (substituída pelas features temporais), colunas de texto livre com baixo valor preditivo, e colunas já combinadas na variável target.

A nona etapa aplica escalonamento robusto às variáveis numéricas usando RobustScaler, que é resistente a outliers e preserva a distribuição dos dados após o tratamento anterior. O escalonamento exclui variáveis identificadoras e a variável target para manter a interpretabilidade.

A décima etapa consolida os dados transformados, resultando em um dataset final com 233.274 registros (após remoção de duplicatas) e 71 features (70 variáveis preditivas + 1 target), consumindo 67,41 MB de memória. O conjunto final apresenta distribuição equilibrada de tipos de dados: 36 variáveis booleanas (one-hot encoded), 28 variáveis float64, 3 variáveis int64, e outras distribuídas entre int16, int8, int32 e float32.

## 3.1.4 Análise exploratória da série temporal

Com os dados devidamente pré-processados e estruturados, foi conduzida uma análise exploratória detalhada para compreender o comportamento histórico da série de faturamento e identificar padrões relevantes para o desenvolvimento dos modelos preditivos. Esta análise foi implementada através de um sistema automatizado de visualizações que gera oito análises específicas da série temporal.

### 3.1.4.1 Visão geral da série temporal

A primeira análise apresenta uma visão geral abrangente da série temporal, incluindo a evolução das vendas ao longo do tempo com linha de tendência, distribuição dos valores por ano através de gráficos de boxplot, e análise das vendas acumuladas. Esta visão panorâmica revelou uma tendência de crescimento consistente de 2014 a 2022, seguida por um declínio significativo em 2023-2024, com valores variando de aproximadamente R$ 8 milhões em 2014 para um pico de R$ 400 milhões em 2022.

### 3.1.4.2 Decomposição STL

A análise de decomposição STL (Seasonal-Trend using Loess) foi aplicada para separar os componentes de tendência, sazonalidade e ruído da série temporal. A decomposição confirmou a presença de uma tendência de longo prazo bem definida e padrões sazonais consistentes, com a série original mostrando crescimento exponencial até 2022, seguido por declínio acentuado. O componente sazonal revelou padrões regulares de variação mensal, enquanto o resíduo indicou períodos de maior volatilidade, especialmente durante os anos de transição.

### 3.1.4.3 Análise de sazonalidade

A análise sazonal detalhada examinou os padrões mensais e de autocorrelação da série temporal. Foram calculadas as médias mensais históricas, revelando que os meses de janeiro, maio e dezembro apresentam consistentemente os maiores volumes de vendas, enquanto fevereiro e junho mostram os menores valores. A análise de autocorrelação identificou dependências temporais significativas até o lag 12, confirmando a presença de sazonalidade anual na série.

### 3.1.4.4 Propriedades estatísticas

A análise das propriedades estatísticas incluiu o cálculo das funções de autocorrelação (ACF) e autocorrelação parcial (PACF), fundamentais para a parametrização de modelos ARIMA. A ACF mostrou correlações significativas nos primeiros lags, decaindo gradualmente, enquanto a PACF apresentou cortes abruptos após os primeiros lags, sugerindo características autorregressivas na série. A análise da série diferenciada (primeira diferença) confirmou a remoção da tendência, tornando a série mais adequada para modelagem estatística.

### 3.1.4.5 Análise de distribuição

A análise de distribuição dos valores de vendas incluiu histograma com sobreposição de distribuição normal, gráfico Q-Q para teste de normalidade, box plot para identificação de outliers, e comparação de densidade entre a distribuição empírica e a normal teórica. Os resultados indicaram que a distribuição das vendas não segue uma distribuição normal, apresentando assimetria positiva e presença de valores extremos, identificados como 3 outliers no box plot.

### 3.1.4.6 Evolução temporal detalhada

A análise de evolução temporal examinou as taxas de crescimento anual, padrões sazonais por ano, e tendência linear geral. O cálculo das taxas de crescimento revelou variações significativas, com crescimento superior a 200% em 2015, estabilização em torno de 20-40% nos anos intermediários, e declínios acentuados de -10% a -50% nos anos finais. A análise de tendência linear mostrou um coeficiente de determinação (R²) de 0,966, indicando que 96,6% da variação dos dados é explicada pela tendência temporal.

### 3.1.4.7 Análise de correlação temporal

A análise de correlação incluiu correlações com lags de 1 a 12 meses, autocorrelação parcial detalhada, matriz de correlação para lags selecionados, e correlação com componentes temporais (ano, trimestre, mês). Os resultados mostraram correlações elevadas (>0,8) para os primeiros lags, decaindo gradualmente até o lag 12. A matriz de correlação dos lags selecionados revelou padrões de dependência temporal que orientaram a configuração dos modelos preditivos.

### 3.1.4.8 Resumo executivo

O resumo executivo consolidou todas as análises anteriores, apresentando características principais dos dados incluindo distribuição por ano, vendas acumuladas com tendência, série temporal com média móvel de 12 meses, distribuição mensal, evolução anual, consistência sazonal através de correlação mensal, e volatilidade anual medida pelo coeficiente de variação. Esta análise revelou alta consistência sazonal (correlações >0,9 entre meses equivalentes) e redução da volatilidade ao longo do tempo (de ~80% para ~20% no coeficiente de variação).

### 3.1.4.9 Insights para modelagem

Com base nesta análise exploratória abrangente, foram identificados os seguintes insights fundamentais para a modelagem preditiva:

- **Estacionariedade**: A série original não é estacionária devido à forte tendência, requerendo diferenciação para modelos ARIMA
- **Sazonalidade**: Presença confirmada de sazonalidade anual (período 12) com padrões consistentes
- **Autocorrelação**: Dependências temporais significativas até 12 lags, orientando a parametrização dos modelos
- **Distribuição**: Dados não seguem distribuição normal, com presença de outliers que requerem tratamento específico
- **Tendência**: Tendência de longo prazo bem definida (R² = 0,966) mas com mudança estrutural após 2022
- **Volatilidade**: Redução da volatilidade ao longo do tempo, indicando maior estabilidade nos padrões recentes

Estes resultados orientaram diretamente a configuração dos parâmetros para cada modelo preditivo, a escolha das técnicas de pré-processamento específicas, e as estratégias de validação temporal adotadas nas etapas subsequentes.

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