# Interaction/Benchmarking Review Report

## Executive Summary

This report reviews the ABISS benchmarking pipeline (the interaction system where a System agent and User agent engage in multi-turn text-to-SQL conversations). The analysis covers 154 interaction JSON files across 7 system agents, 2 datasets (BIRD, Spider), and 3 category-use modes. The review identifies issues in relevance label assignment, ground truth handling, user agent design, and system behavior patterns.

---

## 1. Relevance Label Distribution

### What the data shows

| Label | Count | % |
|-------|-------|---|
| null | 484,953 | 55.2% |
| Relevant | 156,886 | 17.8% |
| Irrelevant | 136,039 | 15.5% |
| Technical | 12,347 | 1.4% |
| **Total interactions** | **790,225** | |

### Null labels are structural, not missing data

The 55.2% null rate initially looks alarming but is entirely expected:

- **142,930** cases: System produced SQL on the first turn (no clarification, no user response needed).
- **169,278** cases: System produced feedback on the first turn (unsolvable question recognized immediately).
- **122,809** cases: System produced SQL on a later turn (terminal interaction, no user response).
- **49,910** cases: System produced feedback on a later turn.
- **26** cases: System produced an empty response (all three fields null). These are genuine edge-case failures.
- **0** cases: System asked a question but relevance is genuinely missing.

**Verdict**: The null labels are not a bug. They mean "no user response was needed at this turn."

---

## 2. Relevance Label Accuracy Issues

### 2.1 "Relevant" label followed by non-resolution

Of **156,886** interactions labeled "Relevant":

| What the system does next | Count | % |
|---------------------------|-------|---|
| Produces SQL (resolves) | 104,847 | 66.8% |
| Asks **another question** | 48,656 | 31.0% |
| Produces feedback (gives up) | 3,374 | 2.2% |
| No next turn | 9 | ~0% |

**31% of the time**, the system receives a relevant answer to its clarification but then asks yet another question instead of producing SQL. In many cases, the system asks the **exact same question it already asked**. This is a system-side failure (failure to incorporate user responses), not a labeling error, but it means the Relevant label was correctly assigned yet wasted.

### 2.2 Answerable questions receiving unnecessary clarification

Of **56,595** answerable+solvable conversations:
- **50,220 (88.7%)**: System correctly produced SQL directly.
- **6,375 (11.3%)**: System asked unnecessary clarification questions.

All unnecessary clarifications were labeled **Irrelevant (6,862)** or **Technical (6,670)**, and **zero Relevant**. This is correct by design (answerable questions cannot receive Relevant labels since there is no semantic ambiguity). However, the solved rate for answerable questions drops from **50.6%** (direct SQL) to **27.5%** (unnecessary clarification), meaning the detour nearly halves accuracy.

All 6,375 cases appear exclusively in `category_use: no_category`, confirming that without category guidance the system over-asks.

### 2.3 Technical label is extremely rare (1.4%)

Only **12,347** out of 790,225 interactions receive the Technical label. This suggests either:
- Systems rarely ask about implementation details (ordering, limits, columns), or
- The user agent conflates Technical with Irrelevant due to missing schema access (see Section 4.2).

---

## 3. Ground Truth Handling Issues

### 3.1 Hidden knowledge leakage into user responses

For **164,472** ambiguous conversations (solvable=true, answerable=false):
- **90,804** user responses closely match the `hidden_knowledge` field (word overlap ratio > 0.4).
- Many have overlap ratio of **1.0**, meaning the user agent essentially parrots the hidden knowledge verbatim.

This makes the user agent unrealistically helpful. A real user would not articulate the disambiguation this precisely. The system gets a nearly perfect answer to its clarification question, yet still only achieves 41.6% execution accuracy at best. This inflates the quality of user responses beyond what a real interaction would produce, potentially masking how poorly systems would perform with less cooperative users.

### 3.2 User agent receives no full GT SQL (by design, but with consequences)

The user agent receives only secondary preferences extracted from GT SQL:
- ORDER BY clauses (raw column names like `T1.enrollment_date DESC`)
- LIMIT values
- DISTINCT flags

**What is NOT extracted** (intentionally, to prevent leakage):
- GROUP BY clauses
- HAVING conditions
- Aggregation functions (SUM vs AVG vs COUNT)
- WHERE conditions
- JOIN conditions

**Consequence**: When the system asks a legitimate implementation question touching GROUP BY, aggregation, or filtering ("Do you want the average or the total?", "Should we group by department or by year?"), the user agent has no GT information to draw on. It defaults to uncertainty ("Either way is fine", "I'm not sure"), which may lead the system to pick the wrong aggregation or grouping. This could be a significant source of wrong final SQLs for questions that depend on these SQL features.

### 3.3 ORDER BY extraction leaks raw column names

The `extract_secondary_preferences` function (`users/sql_preferences.py:24-31`) outputs raw SQL fragments:

```
Results should be ordered by: T1.enrollment_date DESC
```

This exposes table aliases and column names to the user agent, which:
1. Is unnatural (a real user would say "sort by newest first")
2. Could give the system agent schema-level hints it should not have if the user agent parrots these back

### 3.4 Unanswerable questions incorrectly receiving SQL

Of **263,886** unsolvable conversations:
- **211,675 (80.2%)**: Correctly produced feedback.
- **52,199 (19.8%)**: Incorrectly produced SQL for fundamentally unanswerable questions.

Almost all of these are "Missing External Knowledge" or "Missing Schema Elements" questions mispredicted as "Answerable." The system confidently generates SQL for questions that cannot be answered with the available schema/data, with no recognition that something is missing.

Additionally, **50,885** ambiguous questions (solvable but not answerable) were answered with direct SQL and no clarification, skipping the necessary disambiguation step entirely.

---

## 4. Design Issues in the User Agent

### 4.1 `hidden_knowledge` only exists on `QuestionUnanswerable` instances

**Location**: `users/prompts/user_response_prompt.py:138-139`

```python
if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
    prompt += f"**Hidden Knowledge (Your Disambiguating Intent):** {question.hidden_knowledge}\n"
```

The `hidden_knowledge` field only exists on `QuestionUnanswerable` (which covers both ambiguous/solvable and truly unanswerable questions). If any solvable question were ever instantiated as a plain `Question` instead of `QuestionUnanswerable`, the user agent would receive **zero disambiguation information** for a RELEVANT clarification question. It would be forced to hallucinate an answer, potentially misleading the system.

In practice, the current question generation pipeline creates `QuestionUnanswerable` for all non-answerable categories, so this is more of a fragility concern than an active bug. But there is no runtime assertion guarding against it.

### 4.2 User agent has no access to DB schema

**System agent receives**: Full database schema with 5 example rows per table.
**User agent receives**: Nothing about the schema.

This creates an information asymmetry that can cause misclassification:
- If the system asks "Do you want the enrollment date or the registration date?", the user cannot tell whether this is a genuine semantic disambiguation (RELEVANT) or a technical question about which column to use (TECHNICAL).
- The prompt says "Questions asking about columns/tables to use are TECHNICAL" (`user_response_prompt.py:165`), but without schema access the user cannot distinguish between a semantic concept question and a column-name question.

### 4.3 Tie-breaking to IRRELEVANT may discard valid disambiguation

**Location**: `users/user_response.py:106-110`

When models tie (e.g., 1 RELEVANT, 1 TECHNICAL, 1 IRRELEVANT with 3 models), the code defaults to IRRELEVANT. The rationale is conservative ("no info leaks"), but the consequence is that a genuinely relevant clarification question may get shut down with a refusal ("That's not relevant to my question"). The system is then forced to guess the disambiguation, and the resulting SQL is likely wrong.

The fallback at lines 144-147 partially mitigates this: if the tie resolved to IRRELEVANT but no model actually voted IRRELEVANT, all answers are fed to the tournament. But the tournament then evaluates under IRRELEVANT criteria (refusal quality) even though the winning label may be wrong.

### 4.4 Answerable questions can only refuse semantic questions

**Location**: `users/prompts/user_response_prompt.py:180-231`

For answerable questions, only TECHNICAL or IRRELEVANT labels are allowed. If the system mistakenly asks a semantic clarification about an answerable question (a system error, but it happens in 11.3% of cases), the user can only classify it as IRRELEVANT and refuse to answer. A more informative response ("The question is already clear; I mean X") would help the system recover, but the current design provides no recovery path.

---

## 5. Category Misprediction Cascading

### Prediction accuracy

- **216,285** correct category predictions.
- **268,668** incorrect predictions.
- Systems overwhelmingly over-predict "Answerable" (85.4% of predictions).

### Impact on solved rate

| Prediction | Solved Rate |
|------------|------------|
| Correct category | **40.0%** (44,116/110,331) |
| Incorrect category | **11.2%** (18,258/162,935) |
| **Ratio** | **3.6x drop** |

Top mispredictions (all -> Answerable):
- Missing Schema Elements -> Answerable: 66,264
- Structure Ambiguity -> Answerable: 39,978
- Missing Schema Elements -> Missing External Knowledge: 37,230

Category misprediction is the single largest driver of failure. When the system thinks a question is answerable when it is actually ambiguous or unanswerable, it skips clarification and produces wrong SQL or ignores the need for feedback.

---

## 6. Multi-Turn Interaction Patterns

### Conversation length distribution

| Turns | Count |
|-------|-------|
| 1 | 312,225 |
| 2 | 100,038 |
| 3 | 12,836 |
| 4 | 59,854 |

### Relevance degrades across turns

| Turn | Relevant % | Irrelevant % | Technical % |
|------|-----------|-------------|------------|
| 0 | 63.5% | 33.5% | 3.0% |
| 1 | 35.2% | 59.7% | 5.1% |
| 2 | 36.0% | 58.0% | 5.9% |

Systems do not learn from user responses. After the first turn, the relevance rate drops from 63.5% to ~35% and never recovers. For 3+ turn conversations: only **734 improving** vs. **2,262 degrading** (3x more degradation than improvement).

### Repeated questions

**39,280** conversations contain the system asking the exact same question it already asked. Per-agent breakdown:

| Agent | Repeated Q's | % of Multi-turn |
|-------|-------------|-----------------|
| Llama-3.1-8B | 26,212 | 84.4% |
| Qwen2.5-7B | 5,573 | 27.0% |
| gemma-3-27b-it | 2,549 | 7.9% |
| Qwen2.5-32B | 2,055 | 9.3% |
| Qwen2.5-Coder-32B | 2,040 | 8.1% |
| Mistral-Small-3.2-24B | 715 | 3.8% |
| Llama-3.3-70B | 136 | 0.6% |

Llama-3.1-8B has a fundamental context-tracking failure, repeating questions in 84.4% of multi-turn conversations. Larger models (Llama-3.3-70B, Mistral-Small) are far better.

---

## 7. Per-Agent Summary

| Agent | Solved Rate | Relevant % | Repeated Q's | Multi-turn |
|-------|-----------|-----------|-------------|------------|
| Llama-3.1-8B | 14.0% | 47.1% | 26,212 | 31,069 |
| Qwen2.5-7B | 15.2% | 56.6% | 5,573 | 20,617 |
| gemma-3-27b-it | 23.1% | 32.0% | 2,549 | 32,327 |
| Llama-3.3-70B | 27.0% | 67.0% | 136 | 22,803 |
| Mistral-Small-3.2-24B | 27.4% | 74.8% | 715 | 18,687 |
| Qwen2.5-32B | 27.7% | 59.3% | 2,055 | 22,160 |
| Qwen2.5-Coder-32B | 27.8% | 55.7% | 2,040 | 25,065 |

---

## 8. Summary of Issues by Severity

### Critical

| # | Issue | Impact |
|---|-------|--------|
| 1 | **Category misprediction cascading**: 3.6x solved rate drop when wrong | Largest single driver of failure |
| 2 | **System ignores relevant user answers**: 31% of Relevant-labeled interactions followed by another question, not SQL | Wasted disambiguation rounds |
| 3 | **User agent has no schema access**: Cannot distinguish semantic vs. technical questions | Relevance label misclassification |

### Moderate

| # | Issue | Impact |
|---|-------|--------|
| 4 | **Hidden knowledge leakage**: User agent often parrots hidden_knowledge verbatim | Inflates user quality, masks real-world difficulty |
| 5 | **Secondary preferences incomplete**: Missing GROUP BY, HAVING, aggregation info | User defaults to uncertainty for valid technical questions |
| 6 | **Tie-breaking to IRRELEVANT**: Discards valid disambiguation on ties | Forces system to guess, likely producing wrong SQL |
| 7 | **Unanswerable questions get SQL (19.8%)**: System produces SQL for fundamentally unanswerable questions | False confidence, wrong outputs |
| 8 | **Answerable questions can only refuse**: No recovery path when system mistakenly asks semantic questions | System cannot self-correct |

### Low

| # | Issue | Impact |
|---|-------|--------|
| 9 | **ORDER BY extraction leaks column names**: Raw SQL fragments exposed to user agent | Unnatural responses, minor schema leakage |
| 10 | **Repeated questions**: 39,280 conversations, especially Llama-3.1-8B (84.4%) | Wasted interaction turns |
| 11 | **Multi-turn degradation**: Relevance drops from 63.5% to ~35% after turn 0 | Diminishing returns on multi-turn |

---

## 9. Key Takeaways

1. **The null relevance labels (55%) are not a bug.** They are structural: the system produced SQL or feedback directly without asking a clarification question.

2. **The biggest problem is not relevance labeling but the system's failure to act on relevant information.** Even when the system asks a relevant question AND gets a useful answer, it produces SQL only 66.8% of the time. The rest of the time it asks another question or gives up.

3. **Category misprediction is the dominant failure mode.** The 3.6x solved rate gap between correct and incorrect predictions dwarfs all other factors. Systems default to "Answerable" 85% of the time.

4. **The user agent is simultaneously too helpful and too limited.** It leaks hidden knowledge verbatim (too helpful) but cannot distinguish semantic from technical questions due to missing schema access (too limited). The combination creates an unrealistic evaluation surface.

5. **Multi-turn interaction adds little value in the current setup.** Relevance drops sharply after turn 0, many systems repeat questions, and even relevant answers are frequently ignored. The data suggests that if the system cannot resolve the question on the first turn, additional turns are unlikely to help.
