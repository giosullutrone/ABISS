# Taxonomy Comparison: Paper vs Code Implementation

## Analysis Summary

After comparing `taxonomy.tex` with the category implementations, I've identified several areas where the definitions, examples, and explanations could be improved for consistency and clarity.

---

## 🔍 Key Findings

### ✅ **Well-Aligned Categories**

1. **Structural Ambiguity (Scope)** - Definitions match well
2. **Structural Ambiguity (Attachment)** - Definitions match well
3. **Lexical Vagueness** - Definitions match well
4. **Missing External Knowledge** - Definitions match well
5. **Missing User Knowledge** - Definitions match well
6. **Improper Question** - Definitions match well

### ⚠️ **Categories Needing Improvement**

---

## 1. **Semantic Mapping Ambiguity - Lexical Overlap**

### Issues Identified:

**Paper Definition (taxonomy.tex):**
> "Arises when two or more schema attributes **share similar or identical forms**."
> Example: "emails" → `students.personal_email` vs `students.institutional_email`

**Code Definition:**
> "Lexical Overlap arises when two or more schema attributes **share similar or identical forms**, making it unclear which specific variant or type of an attribute a term in the question refers to."

**Problem:** The paper's example is somewhat ambiguous. The word "emails" doesn't literally appear in either column name (`personal_email` vs `institutional_email`). The overlap is in the **concept** "email," not the **lexical form**.

### Recommendation:

**Option A: Strengthen the paper definition** to clarify what "similar forms" means:

```latex
\textit{Lexical Overlap:} Arises when a natural language term maps to two or more 
schema attributes whose names share lexical stems, semantic roots, or refer to 
variants of the same concept. For example, in "List the emails of the students," 
the term "emails" could refer to students.personal_email or students.institutional_email. 
Both columns share the concept of "email" but represent different types.
```

**Option B: Use a clearer example** where the overlap is more literal:

```latex
Example: "What is the date for all projects?" could refer to projects.start_date, 
projects.end_date, or projects.deadline (all containing "date").
```

---

## 2. **Semantic Mapping Ambiguity - Entity Ambiguity**

### Issues Identified:

**Paper Definition:**
> "Occurs when **multiple plausible entities could satisfy the same reference**."

**Code Definition:**
> "Entity Ambiguity occurs when a term or expression in the question can correspond to attributes from **multiple different entities or tables** in the schema."

**Problem:** The code definition is clearer and more precise. The paper's phrase "multiple plausible entities could satisfy the same reference" is vague.

### Recommendation:

**Update paper definition** to match code clarity:

```latex
\textit{Entity Ambiguity:} Occurs when a natural language term or expression 
corresponds to attributes from multiple different entities or tables in the schema. 
The same concept exists across different tables representing different relational 
contexts. For instance, in "List the enrollment date of the students of the 
'database' course," the expression "enrollment date" could refer either to the 
students' university enrollment (students.enrollment_date) or to the enrollment 
in the specific course (student_courses.enrollment_date). The different entity 
interpretations require accessing different tables or following different join paths.
```

---

## 3. **Missing Schema Elements - Missing Entities/Attributes**

### Issues Identified:

**Paper example:**
> "List the administrative staff in the engineering department" - no table for staff

**Code definition:**
> "Missing Entities or Attributes when **key information is completely absent** from the database schema."

**Problem:** The paper example uses "administrative staff" which could be ambiguous. What if there's an "employees" table but it doesn't distinguish staff types? This creates confusion about whether it's:
- Missing entity (no staff table at all)
- Missing attribute (staff table exists but no type/role column)

### Recommendation:

**Clarify the example in the paper**:

```latex
\textit{Missing Entities or Attributes:} Occurs when key information is completely 
absent from the database schema—no table or column could represent the requested data. 
For example, "List the administrative staff in the engineering department" cannot be 
translated into SQL because the schema contains no table representing staff (assuming 
only students, professors, and courses are modeled). Similarly, "Show employee salaries" 
would be unanswerable if an employees table exists but has no salary column.
```

---

## 4. **Missing Schema Elements - Missing Relationships**

### Issues Identified:

**Paper example:**
> "List the professors for the course 'database'" - professors and courses exist but no relationship

This example is good, but it could be strengthened.

### Recommendation:

**Add more context to the paper**:

```latex
\textit{Missing Relationship:} Occurs when the necessary entities exist as separate 
tables but lack foreign keys, junction tables, or other relationship structures to 
connect them. For instance, "List the professors for the course 'database'" cannot 
be answered even though the schema includes both a professors table and a courses 
table, since no relationship (e.g., teaches table with professor_id and course_id) 
links them to represent course assignments.
```

---

## 5. **Conflicting Knowledge**

### Issues Identified:

**Paper Definition:**
> "A question is ambiguous due to Conflicting Knowledge **if the knowledge base K contains multiple, non-equivalent pieces of evidence** for interpreting the same concept."

**Code Definition:**
> "A question is ambiguous due to Conflicting Knowledge when **a hypothetical retrieval system returns multiple, mutually exclusive policies or evidence definitions** for the same concept."

**Problem:** The code definition explicitly mentions "hypothetical retrieval system" which is more realistic for RAG scenarios, but the paper doesn't make this clear. Also, the paper could better emphasize that the ambiguity is NOT in the question itself.

### Recommendation:

**Update paper to match code precision**:

```latex
\subsubsection{Conflicting Knowledge}
A question is ambiguous due to Conflicting Knowledge when the knowledge base K 
contains multiple, mutually exclusive pieces of evidence for interpreting the same 
concept in the question. **Crucially, the ambiguity arises not from the question 
wording itself being vague, but from having multiple documented, conflicting 
interpretations in the knowledge base.** Each piece of evidence provides a valid 
but non-equivalent definition, leading to structurally different SQL queries.

For instance, consider the case where two evidences exist: 
(1) "A student's performance is their average grade" 
(2) "A student's performance is the average grade weighted by the course credits"

Given the request "List the top five students' performance," the ambiguity stems 
from inconsistency within K, not from vagueness in the term "performance" itself. 
The user must specify which policy to follow.
```

---

## 6. **Missing Examples in Code**

### Issues Identified:

**Categories with `get_examples() -> None` in code:**
- MissingSchemaEntitiesCategory
- MissingSchemaRelationshipsCategory

**Problem:** The paper provides examples, but the code returns `None`. This inconsistency makes it harder for LLMs to generate good questions during the automated pipeline.

### Recommendation:

**Add examples to code** to match the paper:

```python
# In missing_schema_entities.py
@staticmethod
def get_examples() -> list[str] | None:
    return [
        "List the administrative staff in the engineering department.",
        "Show employee salaries for the IT department.",
        "Find the office phone numbers for all professors.",
        "What is the research budget for each department?"
    ]

# In missing_schema_relationships.py
@staticmethod
def get_examples() -> list[str] | None:
    return [
        "List the professors for the course 'database'.",
        "Show which students are enrolled in courses taught by Professor Smith.",
        "Find all courses that have prerequisites.",
        "Display departments and their associated research projects."
    ]
```

---

## 7. **Lexical Vagueness - Enhancement Suggestion**

### Current State:

**Paper:**
> "For instance, 'List recent courses' is ambiguous with respect to the temporal threshold defining recent..."

**Code examples:**
```python
"List recent courses.",
"Show employees with high salaries.",
"Find popular products.",
"Display old buildings on campus.",
"Show students with many course enrollments."
```

### Recommendation:

**Add to the paper** a note about different types of vague terms:

```latex
\subsubsection{Lexical Vagueness}
Lexical Vagueness arises when a question contains terms whose meaning lacks a 
precise or objective boundary, leading to indeterminate selection criteria during 
query generation. Such vagueness introduces variability that cannot be resolved 
solely from schema information, as it depends on the user's subjective understanding 
or context-specific conventions.

Common types of vague terms include:
- **Temporal vagueness**: "recent," "old," "soon" (e.g., "List recent courses")
- **Quantitative vagueness**: "many," "few," "high," "low" (e.g., "Show employees with high salaries")
- **Evaluative vagueness**: "popular," "good," "expensive" (e.g., "Find popular products")

For instance, "List recent courses" is ambiguous with respect to the temporal 
threshold defining "recent"—it may refer to the last semester, the last academic 
year, or another undefined interval.
```

---

## 8. **Formal Definitions - Minor Enhancement**

### Current State:

The paper provides excellent formal definitions using set theory and mapping functions.

### Recommendation:

**Add a clarifying note** about the "non-equivalent" criterion:

```latex
\textbf{Definition 1. Mapping Function}
Given a database D and external knowledge K, we define a deterministic function f 
that maps a question q to a subset of S, or to the empty set:
[f(q, D, K) → S_q ⊆ S ∪ {∅}]

The function f(q, D, K) returns the set of queries in S that correctly answer q on D, 
given the disambiguating information in K. If no such query exists, f(q, D, K) = ∅.

**Note:** Two SQL queries are considered non-equivalent if there exists at least one 
database state (possible set of table contents) on which they produce different results. 
This means queries that are syntactically different but semantically equivalent 
(produce the same results on all possible database states) are treated as a single 
element in S.
```

---

## Summary of Recommended Actions

### High Priority (Consistency Issues):

1. ✅ **Add examples to code** for MissingSchemaEntities and MissingSchemaRelationships
2. ✅ **Clarify Lexical Overlap definition** in paper (paper is slightly vague)
3. ✅ **Enhance Entity Ambiguity definition** in paper (code is clearer)
4. ✅ **Improve Conflicting Knowledge** explanation in paper (emphasize ambiguity source)

### Medium Priority (Enhancement):

5. ⚠️ **Add vagueness types** to Lexical Vagueness section in paper
6. ⚠️ **Enhance Missing Schema Elements** examples with more context

### Low Priority (Nice-to-have):

7. 📝 **Add clarifying note** to formal definitions about "non-equivalent"
8. 📝 **Consider adding** a summary table comparing all categories

---

## Suggested Changes to taxonomy.tex

See the attached file `TAXONOMY_IMPROVEMENTS.tex` for specific LaTeX edits.
