# Critical Changes: Side-by-Side Comparison

## 1. Semantic Mapping - Lexical Overlap

### ❌ Current (taxonomy.tex)
```latex
\textit{Lexical Overlap:} Arises when two or more schema attributes share 
similar or identical forms. For example, in "List the emails of the students 
of the 'database' course," the expression "emails" could refer either to the 
students' personal email (students.personal_email) or to the institutional 
one (students.institutional_email). Each interpretation yields a distinct SQL 
mapping.
```

**Problem:** "emails" doesn't literally appear in either column name—overlap is conceptual, not lexical.

### ✅ Improved Version
```latex
\textit{Lexical Overlap:} Arises when a natural language term maps to two or 
more schema attributes whose names share lexical stems, semantic roots, or 
refer to variants of the same concept. For example, in "List the emails of 
the students," the term "emails" could refer to students.personal_email or 
students.institutional_email—both columns share the concept of "email" but 
represent different types or variants. Similarly, "What is the date for all 
projects?" could refer to projects.start_date, projects.end_date, or 
projects.deadline, where all column names contain "date" but refer to distinct 
temporal concepts. Each interpretation yields a distinct SQL mapping.
```

**Why better:** Clarifies both conceptual overlap AND literal lexical overlap with two examples.

---

## 2. Semantic Mapping - Entity Ambiguity

### ❌ Current (taxonomy.tex)
```latex
\textit{Entity Ambiguity:} Occurs when multiple plausible entities could 
satisfy the same reference. For instance, in "List the enrollment date of 
the students of the 'database' course," the expression "enrollment date" 
could refer either to the students' university enrollment 
(students.enrollment_date) or to the enrollment in a specific course 
(student_courses.enrollment_date).
```

**Problem:** "Multiple plausible entities could satisfy the same reference" is vague.

### ✅ Improved Version
```latex
\textit{Entity Ambiguity:} Occurs when a natural language term or expression 
corresponds to attributes from multiple different entities or tables in the 
schema. The same concept exists across different tables representing different 
relational contexts. For instance, in "List the enrollment date of the students 
of the 'database' course," the expression "enrollment date" could refer either 
to the students' university enrollment (students.enrollment_date) or to the 
enrollment in the specific course (student_courses.enrollment_date). The 
different entity interpretations require accessing different tables or following 
different join paths, as they refer to the same concept but in distinct 
relational contexts.
```

**Why better:** Explicitly states it's about SAME CONCEPT in DIFFERENT TABLES/CONTEXTS.

---

## 3. Conflicting Knowledge

### ❌ Current (taxonomy.tex)
```latex
\subsubsection{Conflicting Knowledge}
A question is ambiguous due to Conflicting Knowledge if the knowledge base K 
contains multiple, non-equivalent pieces of evidence for interpreting the same 
concept in the question.

For instance, consider the case where two evidences—"A student's performance 
is their average grade" and "A student's performance is the average grade 
weighted by the course credits"—are present in K. Given the request "List the 
top five students' performance," the ambiguity arises not from the question 
itself but from the inconsistency within K.
```

**Problem:** The key distinguishing fact is buried at the end. Easy to confuse with Lexical Vagueness.

### ✅ Improved Version
```latex
\subsubsection{Conflicting Knowledge}
A question is ambiguous due to Conflicting Knowledge when the knowledge base K 
contains multiple, mutually exclusive pieces of evidence for interpreting the 
same concept in the question. **Crucially, the ambiguity arises not from the 
question wording itself being vague, but from having multiple documented, 
conflicting interpretations in the knowledge base.** Each piece of evidence 
provides a valid but non-equivalent definition, leading to structurally 
different SQL queries. The user must specify which policy or evidence 
interpretation to follow.

For instance, consider the case where two evidences exist in K:
1. "A student's performance is their average grade"
2. "A student's performance is the average grade weighted by the course credits"

Given the request "List the top five students' performance," the ambiguity 
stems from inconsistency within K, not from vagueness in the term "performance" 
itself. The first evidence leads to a simple AVG(grade) calculation, while the 
second requires a weighted average computation. Both are valid interpretations 
according to different policies in K.
```

**Why better:** 
- Emphasizes upfront that question is NOT vague
- Clarifies the source of ambiguity is the KNOWLEDGE BASE
- Better structured with numbered list
- Explains SQL implications

---

## 4. Missing Schema Elements - Entities/Attributes

### ❌ Current (taxonomy.tex)
```latex
\textit{Missing Entities or Attributes:} Occurs when key information is 
absent from the schema. For example, "List the administrative staff in the 
engineering department" cannot be translated into SQL because the schema 
contains no table or column referring to staff.
```

**Problem:** "no table or column" conflates two different scenarios (missing table vs missing column).

### ✅ Improved Version
```latex
\textit{Missing Entities or Attributes:} Occurs when key information is 
completely absent from the database schema—no table or column could represent 
the requested data. For example, "List the administrative staff in the 
engineering department" cannot be translated into SQL if the schema contains 
no table representing staff (assuming only students, professors, and courses 
are modeled). Similarly, "Show employee salaries for the IT department" would 
be unanswerable if an employees table exists but lacks a salary column. The 
missing elements are fundamental entities or attributes that would need to be 
added to the schema structure itself.
```

**Why better:**
- Clarifies "completely absent"
- Gives TWO examples: missing table AND missing column
- Explains what "missing" means (need to add to schema)

---

## 5. Missing Schema Elements - Relationships

### ❌ Current (taxonomy.tex)
```latex
\textit{Missing Relationship:} Occurs when no linkage exists between relevant 
entities. For instance, "List the professors for the course 'database'" cannot 
be answered even though the schema includes both teachers and courses, since 
no relationship links the two tables to represent course assignments.
```

**Problem:** Doesn't explain WHAT a relationship means (foreign key? junction table?).

### ✅ Improved Version
```latex
\textit{Missing Relationship:} Occurs when the necessary entities exist as 
separate tables but lack foreign keys, junction tables, or other relationship 
structures to connect them. For instance, "List the professors for the course 
'database'" cannot be answered even though the schema includes both a 
professors table and a courses table, since no relationship (e.g., a teaches 
junction table with professor_id and course_id foreign keys) links them to 
represent course assignments. Without these relationships, it is impossible 
to construct a SQL query that joins the relevant information together.
```

**Why better:**
- Explicitly mentions "foreign keys, junction tables"
- Gives concrete example of what's missing (teaches table with FKs)
- Explains the consequence (can't construct JOIN)

---

## 6. Lexical Vagueness - Enhancement

### ❌ Current (taxonomy.tex)
```latex
\subsubsection{Lexical Vagueness}
Lexical Vagueness arises when a question contains terms whose meaning lacks 
a precise or objective boundary, leading to indeterminate selection criteria 
during query generation.

For instance, "List recent courses" is ambiguous with respect to the temporal 
threshold defining recent, as it may refer to the last semester, the last 
academic year, or another undefined interval. Such vagueness introduces 
variability that cannot be resolved solely from schema information.
```

**Problem:** Doesn't categorize the different TYPES of vagueness.

### ✅ Improved Version
```latex
\subsubsection{Lexical Vagueness}
Lexical Vagueness arises when a question contains terms whose meaning lacks 
a precise or objective boundary, leading to indeterminate selection criteria 
during query generation. Such vagueness introduces variability that cannot 
be resolved solely from schema information, as it depends on the user's 
subjective understanding or context-specific conventions.

Common types of vague terms include:
- **Temporal vagueness**: "recent," "old," "soon," "earlier" 
  (e.g., "List recent courses")
- **Quantitative vagueness**: "many," "few," "high," "low," "significant" 
  (e.g., "Show employees with high salaries")
- **Evaluative vagueness**: "popular," "good," "expensive," "important" 
  (e.g., "Find popular products")

For instance, "List recent courses" is ambiguous with respect to the temporal 
threshold defining "recent"—it may refer to the last semester, the last 
academic year, or another undefined interval.
```

**Why better:**
- Organizes vagueness into three clear categories
- Provides multiple examples for each type
- Helps readers understand the breadth of lexical vagueness

---

## Summary of Changes

| Category | Change Type | Priority | Impact |
|----------|-------------|----------|--------|
| Lexical Overlap | Clarification | HIGH | Prevents confusion about what "lexical" means |
| Entity Ambiguity | Precision | HIGH | Makes the definition much clearer |
| Conflicting Knowledge | Emphasis | HIGH | Prevents confusion with Lexical Vagueness |
| Missing Entities | Clarification | MEDIUM | Shows both table and column scenarios |
| Missing Relationships | Technical Detail | MEDIUM | Explains what relationships are |
| Lexical Vagueness | Organization | LOW | Nice-to-have categorization |

All improved versions are available in `TAXONOMY_IMPROVEMENTS.tex`.
