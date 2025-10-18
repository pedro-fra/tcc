# Guia Passo a Passo: Configurar Power BI para Comparação XGBoost

## Parte 1: Preparar o Modelo de Dados

### Passo 1.1: Adicionar as Medidas DAX

1. Abra seu arquivo `.pbix` no Power BI Desktop
2. Vá para a aba **"Modelagem"** (ou "Modeling")
3. Clique em **"Nova Medida"** (New Measure)
4. Copie e cole a primeira medida do arquivo `powerbi_medidas_tcc.txt`:

```dax
TCC Valor Faturamento Realizado =
    CALCULATE(
        SUM('[Fato] Faturamento'[VALOR_LIQ]),
        KEEPFILTERS('[Fato] Faturamento'[GERA_COBRANCA] = 1),
        KEEPFILTERS('[Fato] Faturamento'[OPERACAO] = "VENDA")
    )
```

**Importante**: Deixe o nome da medida EXATAMENTE como está

5. Clique em **Enter**
6. Repita os passos 3-5 para TODAS as 8 medidas do arquivo

**Medidas a adicionar:**
- ✓ TCC Valor Faturamento Realizado
- ✓ TCC Faturamento Projetado Simples
- ✓ TCC Diferenca XGBoost vs PowerBI
- ✓ TCC Diferenca Percentual
- ✓ TCC XGBoost Previsao
- ✓ TCC MAE XGBoost
- ✓ TCC RMSE XGBoost
- ✓ TCC MAPE XGBoost

### Passo 1.2: Organizar as Medidas em Pastas

1. Após adicionar todas as medidas, selecione uma delas na lista de campos
2. Na aba **"Modelagem"**, clique em **"Propriedades"**
3. Em **"Pasta de Exibição"**, crie pastas:
   - **TCC\Comparacao** para as medidas 1-4
   - **TCC\Metricas** para as medidas 6-8
   - **TCC\Comparacao** para a medida 5

**Resultado esperado:**
```
[Tabelas]
├── [Fato] Faturamento
├── [Dim] Calendário
└── [Outras...]
├── TCC (pasta de medidas)
    ├── Comparacao
    │   ├── TCC Valor Faturamento Realizado
    │   ├── TCC Faturamento Projetado Simples
    │   ├── TCC Diferenca XGBoost vs PowerBI
    │   ├── TCC Diferenca Percentual
    │   └── TCC XGBoost Previsao
    └── Metricas
        ├── TCC MAE XGBoost
        ├── TCC RMSE XGBoost
        └── TCC MAPE XGBoost
```

---

## Parte 2: Criar Tabela para Exportação

### Passo 2.1: Criar Nova Página no Report

1. Clique em **"+"** embaixo das abas do report para adicionar página
2. Nomeie como **"TCC - Dados Comparacao"**

### Passo 2.2: Criar Tabela com Dados

1. Na página nova, clique em **"Tabela"** no painel de Visualizações
2. Uma tabela será criada no canvas

### Passo 2.3: Adicionar Colunas à Tabela

Na seção **"Valores"** do painel de Campos, arraste:

**Do calendário:**
- `[Dim] Calendário → Data` (para coluna de data)

**Do fato:**
- `[Fato] Faturamento → VALOR_LIQ` (faturamento bruto)

**Das medidas criadas:**
- `TCC → Comparacao → TCC Valor Faturamento Realizado`
- `TCC → Comparacao → TCC Faturamento Projetado Simples`
- `TCC → Comparacao → TCC XGBoost Previsao` (será adicionada depois)

**Resultado esperado:**
```
| Data       | VALOR_LIQ | TCC Valor Faturamento | TCC Faturamento Proj... | TCC XGBoost Previsao |
|------------|-----------|----------------------|------------------------|----------------------|
| out/2014   | ...       | ...                  | ...                    | ...                  |
| nov/2014   | ...       | ...                  | ...                    | ...                  |
...
```

### Passo 2.4: Configurar Formatação da Tabela

1. **Selecione cada coluna de moeda** (as que contêm valores em R$)
2. Na aba **"Formatação"**, altere:
   - **Formato**: Moeda
   - **Unidade de exibição**: Milões (M)
   - **Casas decimais**: 0 ou 2 (conforme preferência)

### Passo 2.5: Adicionar Filtros (Slicers)

1. Clique em **"Segmentador"** no painel de Visualizações
2. Arraste `[Dim] Calendário → Data` para o segmentador
3. Configure como **Entre** (Between) para selecionar período

**Configurar:**
- Escolha período: **Dezembro/2022 até Setembro/2025** (período de teste)

---

## Parte 3: Criar Visuais Comparativos

### Passo 3.1: Gráfico de Linhas - Série Temporal

1. Clique em **"Gráfico de Linhas"**
2. Configure:
   - **Eixo X**: `[Dim] Calendário → MesAno` (ou Data)
   - **Eixo Y**: Adicione as 3 medidas:
     - TCC Valor Faturamento Realizado
     - TCC Faturamento Projetado Simples
     - TCC XGBoost Previsao

3. **Formatação** (clique no pincel):
   - Cores diferentes para cada linha:
     - Realizado: **Preto**
     - Power BI: **Verde**
     - XGBoost: **Azul**
   - Aumentar tamanho das linhas: **3-4px**

4. **Título**: "Comparação: XGBoost vs Power BI vs Real"

**Resultado esperado:**
```
Gráfico mostrando 3 linhas sobrepostas ao longo do tempo
- Linha preta = valores reais
- Linha verde = projeção Power BI
- Linha azul = previsão XGBoost
```

### Passo 3.2: Gráfico de Colunas - Comparação de Diferencas

1. Clique em **"Gráfico de Colunas Agrupadas"**
2. Configure:
   - **Eixo X**: `[Dim] Calendário → MesAno`
   - **Eixo Y**: `TCC → Comparacao → TCC Diferenca XGBoost vs PowerBI`

3. **Formatação**:
   - Cores condicionais:
     - Diferenças positivas: **Azul** (XGBoost maior)
     - Diferenças negativas: **Vermelho** (Power BI maior)

4. **Título**: "Diferença: XGBoost - Power BI (R$)"

### Passo 3.3: Gráfico de Barras - Diferencas Percentuais

1. Clique em **"Gráfico de Barras Agrupadas"**
2. Configure:
   - **Eixo X**: `TCC → Comparacao → TCC Diferenca Percentual`
   - **Eixo Y**: `[Dim] Calendário → MesAno`

3. **Formatação**:
   - Adicionar linha de referência em 0%
   - Cores: Azul para positivo, Vermelho para negativo

4. **Título**: "Diferença Percentual: (XGBoost - PBI) / PBI (%)"

### Passo 3.4: Cards - Métricas de Desempenho

Crie 3 cards para as métricas do XGBoost:

**Card 1 - MAE:**
1. Clique em **"Card"** (Cartão)
2. Arraste: `TCC → Metricas → TCC MAE XGBoost`
3. **Título**: "MAE XGBoost"
4. **Formato**: Moeda em milhões

**Card 2 - RMSE:**
1. Repita para `TCC → Metricas → TCC RMSE XGBoost`

**Card 3 - MAPE:**
1. Repita para `TCC → Metricas → TCC MAPE XGBoost`
2. **Formato**: Percentual com 2 casas decimais

**Posicione os cards no topo da página**

---

## Parte 4: Configurar Layout e Filtros

### Passo 4.1: Organizar Layout da Página

1. **Página 1 (Visão Geral)**:
   - Linha 1: 3 Cards (MAE, RMSE, MAPE)
   - Linha 2: Gráfico de linhas (série temporal)
   - Linha 3: Segmentador de data

2. **Página 2 (Análise Detalhada)**:
   - Topo: Segmentador de data
   - Esquerda: Gráfico de colunas (diferenças R$)
   - Direita: Gráfico de barras (diferenças %)
   - Embaixo: Tabela de dados

### Passo 4.2: Adicionar Filtros Globais

1. No painel **"Filtros"** (lado esquerdo), clique em **"Adicionar Filtro"**
2. Selecione `[Dim] Calendário → Data`
3. Configure como:
   - **Tipo de filtro**: Entre (Advanced)
   - **Data Inicial**: 01/12/2022 (Dezembro/2022)
   - **Data Final**: 30/09/2025 (Setembro/2025)
   - Clique em **"Aplicar Filtro"**

### Passo 4.3: Sincronizar Filtros

1. Para cada visual que tem data como eixo:
   - Clique com botão direito → **"Sincronizar Filtros"**
   - Marque as páginas onde ele deve filtrar

---

## Parte 5: Exportar Dados para CSV

### Passo 5.1: Exportar Tabela de Dados

1. **Clique com botão direito** na tabela que criou no Passo 2.2
2. Selecione **"Exportar dados"** (Export data)
3. Escolha **"CSV"** como formato
4. Salve como: **`powerbi_historico_test_period.csv`** em `data/`

**Caminho esperado:**
```
c:\Users\PedroFrá\Documents\coding\tcc\data\powerbi_historico_test_period.csv
```

### Passo 5.2: Verificar Estrutura do CSV

Abra o arquivo e verifique que tem as colunas:
```
Data, VALOR_LIQ, TCC Valor Faturamento Realizado, TCC Faturamento Projetado Simples, TCC XGBoost Previsao
```

---

## Parte 6: Adicionar Coluna XGBoost (Quando Disponível)

**Nota**: Estes passos são para DEPOIS que você executar o script Python

### Passo 6.1: Importar Resultados do XGBoost

1. Clique em **"Obter Dados"** (Get Data)
2. Selecione **"Texto/CSV"**
3. Navegue até o arquivo gerado pelo Python com previsões do XGBoost
4. Carregue os dados

**O arquivo terá:**
```
Data, Previsao_XGBoost
2022-12-31, 50000000
2023-01-31, 55000000
...
```

### Passo 6.2: Mesclar com Tabela Principal

1. Em **"Modelagem"**, clique em **"Nova Tabela"**
2. Use DAX para mesclar os dados:

```dax
TCC_XGBoost_Previsoes =
    SELECTCOLUMNS(
        [Tabela XGBoost],
        "Data", [Data],
        "Previsao", [Previsao_XGBoost]
    )
```

3. Crie relacionamento:
   - De: `TCC_XGBoost_Previsoes[Data]`
   - Para: `[Dim] Calendário[Data]`

### Passo 6.3: Atualizar Medida de XGBoost

Edite a medida `TCC XGBoost Previsao`:

```dax
TCC XGBoost Previsao =
    CALCULATE(
        SUM('TCC_XGBoost_Previsoes'[Previsao]),
        KEEPFILTERS('TCC_XGBoost_Previsoes'[Data] = MAX('[Dim] Calendário'[Data]))
    )
```

---

## Parte 7: Testar e Validar

### Passo 7.1: Verificações Visuais

- [ ] Tabela mostra 27 meses (dez/2022 a set/2025)
- [ ] Todos os valores estão no intervalo esperado
- [ ] Os 3 cards mostram métricas do XGBoost
- [ ] Gráfico de linhas mostra 3 séries diferentes
- [ ] Diferenças são calculadas corretamente

### Passo 7.2: Validação de Dados

1. Selecione um mês específico na tabela
2. Verifique:
   - Faturamento Realizado = SUM dos dias do mês
   - XGBoost Previsão = valor da previsão do modelo
   - Diferença = XGBoost - Power BI

### Passo 7.3: Exportar Final

1. Depois que tudo está validado
2. Clique com direito na tabela → **"Exportar dados"**
3. Salve como `powerbi_historico_test_period.csv`

---

## Parte 8: Executar Script Python

### Passo 8.1: Rodar Comparação

```bash
cd c:\Users\PedroFrá\Documents\coding\tcc
uv run python run_comparison_powerbi.py data/powerbi_historico_test_period.csv
```

### Passo 8.2: Verificar Saídas

Verifique se foram criados:
- ✓ `COMPARACAO_XGBOOST_POWERBI.md` (atualizado com dados)
- ✓ `data/plots/powerbi_comparison/` (4 gráficos PNG)
- ✓ `data/processed_data/comparison_results.json` (dados JSON)

---

## Dicas Importantes

### Dica 1: Formatação de Valores

Para todas as medidas que representam moeda:
1. Selecione a medida
2. Na aba **"Modelagem"** → **"Formatação"**
3. Defina como:
   - Formato: Moeda (R$)
   - Casas decimais: 0 (para milhões)

### Dica 2: Ordem de Meses

Se os meses não aparecerem em ordem:
1. Clique em **"MesAno"** na tabela
2. Menu **"Classificar"** → **"Classificar por outro coluna"**
3. Selecione **"AnoMesINT"** (formato YYYYMM)

### Dica 3: Segmentador de Data

Para facilitar seleção:
1. No segmentador, clique no ícone de funil
2. Escolha **"Entre"** como tipo
3. Configure datas manuais:
   - De: 01/12/2022
   - Até: 30/09/2025

### Dica 4: Sincronizar Filtros

Cada visual precisa estar sincronizado com o segmentador:
1. Clique no visual
2. **Formatar** → **Interações** → **Editar interações**
3. Certifique-se que o segmentador filtra o visual

### Dica 5: Cores Consistentes

Use as mesmas cores em todos os gráficos:
- **Real/Faturamento**: Preto ou Cinza escuro
- **Power BI**: Verde
- **XGBoost**: Azul

---

## Checklist Final

Antes de exportar dados:

- [ ] Todas as 8 medidas foram adicionadas
- [ ] Medidas estão organizadas em pastas
- [ ] Página "TCC - Dados Comparacao" foi criada
- [ ] Tabela contém as 5 colunas esperadas
- [ ] Período está filtrado (dez/2022 - set/2025)
- [ ] Cards mostram valores do XGBoost
- [ ] Gráfico de linhas mostra 3 séries
- [ ] Diferenças estão calculadas
- [ ] CSV foi exportado corretamente
- [ ] CSV contém 27 linhas (27 meses)

---

## Troubleshooting

### Problema: Medida retorna em branco

**Solução**:
- Verifique se está dentro do período de dados
- Verifique se os filtros GERA_COBRANCA=1 e OPERACAO="VENDA" estão presentes

### Problema: Valores não correspondem com histórico

**Solução**:
- Confirme que está usando DATE_CAD, não DATA_EMISSAO_PEDIDO
- Verifique os filtros na medida `TCC Valor Faturamento Realizado`

### Problema: CSV não exporta corretamente

**Solução**:
- Certifique-se que a tabela está visível (não contraída)
- Tente exportar via Power BI Online em vez de Desktop
- Verifique permissões de escrita na pasta `data/`

---

## Próximos Passos

Depois de exportar o CSV:

1. Execute o script Python:
   ```bash
   uv run python run_comparison_powerbi.py data/powerbi_historico_test_period.csv
   ```

2. Verifique os resultados em:
   - `COMPARACAO_XGBOOST_POWERBI.md`
   - `data/plots/powerbi_comparison/`

3. Documente as conclusões no TCC!

---

**Tempo estimado**: 1-2 horas

**Requisitos**: Power BI Desktop 2.0+
