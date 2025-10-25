# Análise da Efetividade do R² para o Projeto TCC

**Data:** 25 de outubro de 2025

---

## RESPOSTA DIRETA

**NÃO, R² NÃO É EFETIVO PARA ESTE PROJETO**

- **R² Calculado:** 0.5652 (56.52%)
- **Interpretação:** Modelo explica 56.52% da variância dos dados
- **Problema:** Essa métrica mascara a qualidade real das previsões em série temporal

---

## 1. Cálculo do R² para seu Projeto

### Fórmula
```
R² = 1 - (SS_res / SS_tot)
onde:
  SS_res = Σ(Real - Previsão)²  = 347,668,475,577
  SS_tot = Σ(Real - Média)²     = 799,517,111,975
```

### Resultado
```
R² = 1 - (347,668,475,577 / 799,517,111,975)
R² = 0.5652 (56.52%)
```

### Interpretação Padrão
- **< 30%:** Ajuste ruim
- **30-50%:** Moderado
- **50-70%:** **Bom** ← Seu projeto está aqui
- **70-90%:** Muito bom
- **> 90%:** Excelente

---

## 2. Por que R² NÃO é Efetivo para Série Temporal

### Problema 1: Sazonalidade Inflaciona R²
Seus dados têm forte sazonalidade (ver gráficos):
- Picos em dezembro e janeiro
- Vales em abril-junho
- R² sobe automaticamente quando ambos (real e previsão) seguem o mesmo padrão sazonal

### Problema 2: Tendência Disfarça Maus Modelos
```
Exemplo:
  Se REAL segue: 1000 → 1100 → 1200 → 1300
  E PREVISTO segue: 900 → 1000 → 1100 → 1200
  R² será ALTO mesmo sendo sempre -100 de erro!
```

### Problema 3: Não Penaliza Bias
R² não mostra se o modelo:
- Sempre superestima (previsões maiores)
- Sempre subestima (previsões menores)

Você tem **Viés de 1.47%** (leve superestimação), mas R² não captura isso.

### Problema 4: Escala Dependente
R² varia conforme a magnitude dos dados:
- Dados em milhões → R² diferente
- Dados em bilhões → R² diferente

MAPE não tem esse problema (sempre em %)

### Problema 5: Baseline Inadequado
R² usa a média aritmética como baseline:
```
SS_tot = Σ(Real - Média)²
```

Para série temporal, a média não é um baseline apropriado. Seria melhor comparar com:
- Previsão naive (valor anterior)
- Média móvel
- Sazonalidade pura

---

## 3. Comparação: R² vs MAPE

| Aspecto | R² (56.52%) | MAPE (23.45%) |
|---------|------------|---------------|
| **Valor** | 0.5652 | 23.45% |
| **Interpretação** | Variância explicada | Erro médio em % |
| **Escala** | Dependente dos dados | Sempre em % |
| **Intuitivo** | Técnico | Negócio |
| **Para série temporal** | Problemático | Excelente |
| **Comunica ao stakeholder** | Difícil | Fácil |

### Exemplo Prático
```
MAPE (23.45%):
"Em média, a previsão erra 23.45% em relação ao real"
└─ Qualquer pessoa entende

R² (56.52%):
"O modelo explica 56.52% da variância dos dados"
└─ Precisa ser técnico para entender
```

---

## 4. Métricas Já Calculadas no Seu Projeto

Você JÁ TEM as melhores métricas para série temporal:

| Métrica | Valor | O que Mede |
|---------|-------|-----------|
| **MAPE** | 23.45% | Erro percentual médio |
| **MAE** | R$ 89,651 | Erro absoluto em reais |
| **RMSE** | R$ 119,786 | Raiz do erro quadrático (penaliza grandes erros) |
| **Viés** | 1.47% | Tendência de super/subestimar |
| **Acurácia** | 79.44% | Acerto comparado à média |

✓ **Essas métricas são suficientes e superiores a R²**

---

## 5. Métricas Adicionais Opcionais

Se quiser complementar, considere:

### MASE (Mean Absolute Scaled Error)
```
Compara seu modelo com "previsão naive"
(simplesmente usar o valor do mês anterior)

Para seu projeto:
  MASE = 0.5848 (< 1)
  └─ Seu modelo é MELHOR que naive
```

**Quando usar:** Ideal para comparar com baseline simples

### Direcionalidade (Direction Accuracy)
```
Percentual de vezes que a previsão acerta a "direção"
(se é maior ou menor que mês anterior)

Para seu projeto:
  Direcionalidade = 72.0%
  └─ Acerta tendência 72% das vezes
```

**Quando usar:** Comunicar acerto de tendências

---

## 6. Recomendação Final para sua Monografia

### O QUE USAR:

**Métricas Principais (obrigatório):**
- MAPE: 23.45% ← **Principal, para comparar com XGBoost**
- MAE: R$ 89,651
- RMSE: R$ 119,786

**Complementares (recomendado):**
- Viés: 1.47%
- Acurácia: 79.44%

### O QUE NÃO USAR:

**Evitar:**
- ~~R² (56.52%)~~ ← Inadequado para série temporal
- ~~RMSE em R$~~ ← Use MAE para melhor interpretação

### O QUE OPCIONALMENTE ADICIONAR:

**Se quiser mais rigor:**
- MASE (compara com naive)
- Direcionalidade (acerto de tendências)
- Gráficos de resíduos por período

---

## 7. Conclusão Executiva

**Para responder:** "A métrica de R² é efetiva para este projeto?"

**Resposta:** **NÃO**

**Motivos:**
1. ✗ Dados têm forte sazonalidade (R² inflacionado)
2. ✗ MAPE (23.45%) é muito mais apropriado
3. ✗ R² mascara problemas em períodos específicos
4. ✓ Você já tem as melhores métricas (MAPE, MAE, RMSE, Viés, Acurácia)

**Recomendação:**
- Use MAPE como métrica principal na monografia
- Mantenha MAE, RMSE, Viés, Acurácia
- Não mencione R² em série temporal de faturamento

---

## 8. Comparação com XGBoost

| Métrica | Power BI Híbrido | XGBoost | Melhor |
|---------|-----------------|---------|--------|
| MAPE | 23.45% | 26.91% | Power BI 12.9% |
| MAE | R$ 89,651 | R$ 10.1M | Power BI 99.1% |
| RMSE | R$ 119,786 | R$ 13.3M | Power BI 99.1% |
| R² | 0.5652 | ? | N/A |

✓ Power BI (Método Híbrido) é claramente melhor em TODAS as métricas apropriadas

---

## Referências

Para série temporal de previsão:
- **Melhor métrica:** MAPE (Mean Absolute Percentage Error)
- **Razão:** Escala independente, fácil de comunicar, apropriada para séries
- **Evitar:** R², correlação, regressão linear padrão

