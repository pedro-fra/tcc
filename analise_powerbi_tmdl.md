# Análise do Código TMDL do Power BI - Faturamento e Calendário

## Sumário Executivo

Este documento documenta a análise do arquivo `Factory_TMDL.txt` (Tabular Model Definition Language do Power BI), focando nas medidas de faturamento e relacionamentos entre as tabelas de Faturamento (`[Fato] Faturamento`) e Calendário (`[Dim] Calendário`).

---

## 1. Tabela de Faturamento - `[Fato] Faturamento`

### Localização no TMDL
- **Nome da Tabela**: `[Fato] Faturamento`
- **LineageTag**: `862368ab-5be9-4267-9a1d-18de62b9f17d`
- **Tipo**: Tabela Fato (Fact Table)

### Colunas Principais para Análise

#### 1.1 Coluna de Valor (Target Variable)
- **Nome**: `VALOR_LIQ`
- **Tipo de Dados**: Decimal (Currency)
- **Formato**: R$ (Reais Brasileiros)
- **Descrição**: Valor líquido de cada transação de faturamento
- **Sumarização Padrão**: SUM
- **Nota**: Esta é a coluna que contém os valores de vendas utilizados nos cálculos de faturamento

#### 1.2 Coluna de Data Principal
- **Nome**: `DATE_CAD`
- **Tipo de Dados**: DateTime
- **Descrição**: Data de cadastro/criação do registro
- **Relacionamento Ativo**: RELACIONADO com `[Dim] Calendário`.Data (RELAÇÃO ATIVA ID: `3e3b5424-e320-472f-a9e8-bc827f90c661`)

#### 1.3 Coluna de Data Alternativa
- **Nome**: `DATA_EMISSAO_PEDIDO`
- **Tipo de Dados**: DateTime (Date)
- **Descrição**: Data de emissão do pedido original
- **Relacionamento**: RELACIONADO com `[Dim] Calendário`.Data (RELAÇÃO INATIVA ID: `8c4ac02e-bba4-8a1a-4235-717ee68eb7d9`)
- **Status**: INATIVA (não é usada por padrão)

#### 1.4 Coluna de Tipo de Operação
- **Nome**: `OPERACAO`
- **Tipo de Dados**: String
- **Descrição**: Tipo de operação comercial (VENDA, VENDA DE MERCADORIA CONSIGNADA, etc.)
- **Nota**: Usada em filtros para selecionar apenas operações de venda

#### 1.5 Coluna de Cobrança
- **Nome**: `GERA_COBRANCA`
- **Tipo de Dados**: Int64 (booleano)
- **Descrição**: Indicador se a transação gera cobrança (1 = sim, 0 = não)
- **Nota**: Usada em filtros para selecionar apenas transações que geram cobrança

#### 1.6 Outras Colunas Relevantes
- `ID_CLIENTE`: Identificador do cliente (relacionamento com `[Dim] Cliente`)
- `BF_ID_VENDEDOR`: Identificador do vendedor (relacionamento com `[Dim] Vendedores`)
- `CLIENTE_SEGMENTO`: Segmento do cliente (relacionamento com `[Dim] Segmento`)
- `ID_FATURAMENTO`: Chave primária da tabela

---

## 2. Tabela de Calendário - `[Dim] Calendário`

### Localização no TMDL
- **Nome da Tabela**: `[Dim] Calendário`
- **LineageTag**: `8e99c5de-06f6-4f6b-9658-114ac62b1bb2`
- **Tipo**: Tabela Dimensão (Dimension Table)
- **DataCategory**: Time

### Colunas Principais

#### 2.1 Coluna Chave (Primary Key)
- **Nome**: `Data`
- **Tipo de Dados**: DateTime (Date)
- **isKey**: true
- **Descrição**: Data única da dimensão calendário (chave primária)

#### 2.2 Colunas de Período

**Ano e Mês:**
- `Ano`: Ano da data (int64)
- `NomeMes`: Nome do mês (string, ex: "Janeiro", "Fevereiro", etc.)
- `MesNum`: Número do mês (1-12)
- `MesAno`: String combinada (ex: "Jan/2025")
- `AnoMesINT`: Integer combinado ano-mês para ordenação

**Períodos Maiores:**
- `Trimestre`: Trimestre do ano (1-4)
- `TrimestreAbreviado`: Abreviação do trimestre (ex: "T1")
- `Bimestre`: Bimestre do ano
- `Semestre`: Semestre do ano (1-2)

#### 2.3 Colunas de Dia

- `Semana`: Número da semana do ano
- `DiaSemana`: Dia da semana (1=segunda, 7=domingo, etc.)
- `NomeDia`: Nome do dia (SEGUNDA, TERÇA, QUARTA, QUINTA, SEXTA, SÁBADO, DOMINGO)

#### 2.4 Colunas Calculadas para Faturamento

- **`Dia_Filtro`** (calculada): Filtra dias do mês atual que já passaram
  - Retorna "Ok" se o dia já passou no mês atual
  - Usada para cálculos de projeção de faturamento

- **`BF_Mês Atual`** (presumida): Campo binário para identificar "Mês Atual"
- **`BF_Dia`** (presumida): Campo para identificar dias
- **`BF_Dia_Filtro`** (presumida): Campo de filtro para dias válidos
- **`Feriado`** (presumida): Indicador de feriado (Sim/Não)

#### 2.5 Outras Colunas
- `BF_FiltroMesmoPeriodo`: Relacionamento com tabela auxiliar `Aux_MesmoPeriodo`

---

## 3. Relacionamentos entre Faturamento e Calendário

### Relacionamento Ativo Principal

**ID**: `3e3b5424-e320-472f-a9e8-bc827f90c661`
```
From: [Fato] Faturamento.DATE_CAD
To:   [Dim] Calendário.Data
Status: ATIVA (padrão)
Cardinalidade: Many-to-One
```

**Descrição**:
- A data de cadastro (`DATE_CAD`) da tabela de Faturamento é relacionada com a dimensão de Calendário
- Este é o relacionamento usado por padrão em todos os cálculos
- Permite filtrar e agrupar faturamentos por data de cadastro

**Impacto**:
- Todos os filtros de data aplicados via Calendário afetam os valores de faturamento
- Agregações por mês, trimestre, ano, etc., são baseadas neste relacionamento

### Relacionamento Inativo Alternativo

**ID**: `8c4ac02e-bba4-8a1a-4235-717ee68eb7d9`
```
From: [Fato] Faturamento.DATA_EMISSAO_PEDIDO
To:   [Dim] Calendário.Data
Status: INATIVA
Cardinalidade: Many-to-One
```

**Descrição**:
- A data de emissão do pedido (`DATA_EMISSAO_PEDIDO`) poderia ser usada como alternativa
- Está desativada por padrão (não é usada automaticamente)
- Pode ser ativada manualmente em cálculos específicos quando necessário

**Diferença**:
- `DATE_CAD` é a data de cadastro do pedido no sistema
- `DATA_EMISSAO_PEDIDO` é a data original de emissão do pedido

---

## 4. Medida Principal: "Totalizador Faturamento Projetado"

### Definição DAX

```dax
Totalizador Faturamento Projetado =
    SUMX(
        VALUES('[Dim] Calendário'[BF_Mês Atual]),
        [Faturamento Projetado]
    )
```

### Componentes

#### 4.1 Medida Base: "Faturamento Projetado"

```dax
Faturamento Projetado =
    VAR vDiasUteis =
    CALCULATE(
        COUNTROWS('[Dim] Calendário'),
        KEEPFILTERS('[Dim] Calendário'[NomeDia] <> "DOMINGO"),
        KEEPFILTERS('[Dim] Calendário'[NomeDia] <> "SÁBADO"),
        KEEPFILTERS('[Dim] Calendário'[BF_Mês Atual] = "Mês Atual"),
        KEEPFILTERS('[Dim] Calendário'[Feriado] = "Não"),
        ALL(Aux_MesmoPeriodo)
    )

    VAR DiasAteHoje =
    CALCULATE(
        DISTINCTCOUNT('[Dim] Calendário'[BF_Dia]),
        KEEPFILTERS('[Dim] Calendário'[BF_Dia] <= DAY(TODAY())),
        KEEPFILTERS('[Dim] Calendário'[BF_Dia_Filtro] = "OK"),
        KEEPFILTERS('[Dim] Calendário'[BF_Mês Atual] = "Mês Atual"),
        KEEPFILTERS('[Dim] Calendário'[Feriado] = "Não"),
        ALL(Aux_MesmoPeriodo)
    )

    RETURN
    SWITCH(
        TRUE(),
        SELECTEDVALUE('[Dim] Calendário'[BF_Mês Atual]) = "Mês Atual",
        DIVIDE([Valor Faturamento], DiasAteHoje) * vDiasUteis,
        [Valor Faturamento]
    )
```

#### 4.2 Medida de Faturamento Realizado: "Valor Faturamento"

```dax
Valor Faturamento =
    CALCULATE(
        SUM('[Fato] Faturamento'[VALOR_LIQ]),
        KEEPFILTERS('[Fato] Faturamento'[GERA_COBRANCA] = 1),
        KEEPFILTERS('[Fato] Faturamento'[OPERACAO] IN {"VENDA", "VENDA DE MERCADORIA CONSIGNADA"})
    )
```

### Lógica de Funcionamento

1. **Para Mês Completo**: Retorna o valor realizado de `Valor Faturamento`
2. **Para Mês Atual (em andamento)**:
   - Calcula o faturamento médio por dia útil
   - Multiplica pela quantidade de dias úteis do mês completo
   - Resultado: Projeção de faturamento total esperado para o mês

### Cálculo Detalhado

```
Faturamento Projetado (Mês Atual) =
    (Faturamento até hoje / Dias úteis transcorridos) × Dias úteis totais do mês
```

**Exclusões**:
- Sábados e domingos não são contados como dias úteis
- Feriados não são contados como dias úteis (coluna `Feriado = "Não"`)

---

## 5. Filtros Aplicados nas Medidas

### Filtros em "Valor Faturamento"

```
GERA_COBRANCA = 1 (transação gera cobrança)
OPERACAO IN {"VENDA", "VENDA DE MERCADORIA CONSIGNADA"}
```

### Filtros em "Faturamento Projetado"

```
NomeDia <> "DOMINGO"
NomeDia <> "SÁBADO"
BF_Mês Atual = "Mês Atual"
Feriado = "Não"
```

---

## 6. Estrutura de Dados para o Projeto TCC

### Dados Disponíveis para Exportação

**Tabela Base**: `[Fato] Faturamento`

**Colunas Necessárias**:
1. `DATE_CAD` - Data para agregação mensal
2. `VALOR_LIQ` - Valor de faturamento
3. `OPERACAO` - Tipo de operação (filtro)
4. `GERA_COBRANCA` - Indicador de cobrança (filtro)
5. `ID_CLIENTE` - Para identificação de cliente (anonimizar)

**Período Coberto**: Todos os registros históricos

**Filtros Recomendados**:
- `GERA_COBRANCA = 1`
- `OPERACAO = "VENDA"` (ou incluir "VENDA DE MERCADORIA CONSIGNADA")

### Agregação Sugerida

Os dados originais devem ser agregados mensalmente:

```sql
SELECT
    EOMONTH(DATE_CAD) AS Mes,
    SUM(VALOR_LIQ) AS Faturamento_Mensal,
    COUNT(DISTINCT ID_CLIENTE) AS Num_Clientes,
    COUNT(*) AS Num_Transacoes
FROM [Fato] Faturamento
WHERE GERA_COBRANCA = 1
  AND OPERACAO = 'VENDA'
GROUP BY EOMONTH(DATE_CAD)
ORDER BY Mes
```

---

## 7. Nota Importante: Alinhamento com Metodologia do TCC

O relacionamento `[Fato] Faturamento.DATE_CAD` → `[Dim] Calendário.Data` é o mesmo relacionamento utilizado para:

1. **Cálculo da projeção mensal** no Power BI
2. **Agregação temporal** dos dados
3. **Cálculo de dias úteis** para projeção

Ao importar dados do Power BI para análise de ML/previsão, deve-se:

- Usar `DATE_CAD` como base para agregação mensal (alinhado com Power BI)
- Manter filtros de `GERA_COBRANCA = 1` e `OPERACAO = "VENDA"`
- Agrupar por mês completo (EOMONTH)
- Anonimizar dados de cliente conforme protocolo de privacidade

---

## 8. Conclusão

O modelo de dados do Power BI está bem estruturado para análise de faturamento, com:

✓ Relacionamento claro entre faturamento e calendário
✓ Medidas consolidadas e reutilizáveis
✓ Filtros apropriados para dados comerciais
✓ Lógica de projeção robusta baseada em dias úteis

Esta documentação serve como referência para integração dos modelos de ML desenvolvidos neste TCC com o sistema Power BI existente.
