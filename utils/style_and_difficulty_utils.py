from dataset_dataclasses.question import QuestionDifficulty, QuestionStyle


# SQL Difficulty Criteria Dictionary
DIFFICULTY_CRITERIA = {
    QuestionDifficulty.SIMPLE: '''**Simple SQL Criteria:**
Simple SQL queries satisfy one or more of the following:
- Select data from a single table only
- Use basic filtering with WHERE clauses
- Apply simple sorting with ORDER BY
- Include basic aggregate functions (COUNT, SUM, AVG, MIN, MAX) without GROUP BY
- No joins, subqueries, or advanced SQL features

**Example:**
```sql
SELECT name, department_name
FROM employees
WHERE level > 5
ORDER BY age DESC;
```''',

    QuestionDifficulty.MODERATE: '''**Moderate SQL Criteria:**
Moderate SQL queries satisfy one or more of the following:
- Use table joins (INNER JOIN, LEFT JOIN, RIGHT JOIN, CROSS JOIN)
- Include subqueries in SELECT or WHERE clauses
- Combine aggregate functions with GROUP BY
- Use complex WHERE conditions (IN, BETWEEN, LIKE, multiple AND/OR)
- Apply HAVING clauses to filter grouped results
- Involve multiple aggregate functions

**Example:**
```sql
SELECT e.name, d.department_name, AVG(s.salary) AS average_salary
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id
LEFT JOIN salaries s ON e.employee_id = s.employee_id
WHERE e.age > 30 AND e.status = 'active'
GROUP BY e.name, d.department_name
HAVING AVG(s.salary) > 50000;
```''',

    QuestionDifficulty.COMPLEX: '''**Complex SQL Criteria:**
Complex SQL queries satisfy one or more of the following:
- Use nested subqueries with multiple levels
- Apply multiple types of joins, including self-joins
- Implement window functions (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD)
- Use Common Table Expressions (CTEs) for query organization
- Combine multiple aggregate functions with complex grouping
- Include complex WHERE and HAVING clauses with nested conditions
- Apply advanced SQL functions and operators

**Example:**
```sql
WITH EmployeeCTE AS (
    SELECT employee_id, name, department_id, 
           ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank
    FROM employees
)
SELECT e.name, d.department_name
FROM EmployeeCTE e
INNER JOIN departments d ON e.department_id = d.department_id
WHERE e.rank <= 3;
```''',

    QuestionDifficulty.HIGHLY_COMPLEX: '''**Highly Complex SQL Criteria:**
Highly complex SQL queries satisfy one or more of the following:
- Use multiple CTEs with interdependencies
- Implement recursive CTEs for hierarchical data traversal
- Combine nested subqueries with various join types
- Apply extensive window functions with complex partitioning
- Use UNION/UNION ALL to combine multiple result sets
- Implement advanced analytical functions and complex aggregations
- Employ a wide range of SQL clauses with intricate logic
- Utilize advanced SQL features like PIVOT, lateral joins, or array operations

**Example:**
```sql
WITH RECURSIVE EmployeeHierarchy AS (
    SELECT employee_id, name, manager_id, department_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    SELECT e.employee_id, e.name, e.manager_id, e.department_id, eh.level + 1
    FROM employees e
    JOIN EmployeeHierarchy eh ON e.manager_id = eh.employee_id
),
DepartmentSalaries AS (
    SELECT eh.employee_id, eh.name, eh.level, d.department_name, s.salary, d.department_id
    FROM EmployeeHierarchy eh
    INNER JOIN departments d ON eh.department_id = d.department_id
    INNER JOIN salaries s ON eh.employee_id = s.employee_id
),
DepartmentStats AS (
    SELECT 
        d.department_id,
        COUNT(e.employee_id) AS employee_count,
        AVG(s.salary) AS average_salary
    FROM employees e
    INNER JOIN salaries s ON e.employee_id = s.employee_id
    INNER JOIN departments d ON e.department_id = d.department_id
    GROUP BY d.department_id
)
SELECT ds.name, ds.level, 
       SUM(ds.salary) OVER (PARTITION BY ds.department_id ORDER BY ds.level, ds.name) AS cumulative_salary
FROM DepartmentSalaries ds
INNER JOIN DepartmentStats dstat ON ds.department_id = dstat.department_id
ORDER BY ds.level, ds.name;
```'''
}


"""
From OMNI-SQL paper:

"Formal": '''**Formal Style**
   - Uses standard grammar and vocabulary.
   - Example: Find all students older than 18 years and return their home addresses.''',

"Colloquial": '''**Colloquial Style**
   - Employs informal vocabulary and expressions.
   - Example: Hey! Could you help me find all the students who are over 18? I'd love to know their names and where they live.''',

"Imperative": '''**Imperative Style**
   - Uses command or directive sentences.
   - Example: Could you please gather all the students who are older than 18? I really need to know their names and where they live!''',

"Interrogative": '''**Interrogative Style**
   - Uses question forms.
   - Example: Could you tell me which students are older than 18 and what their home addresses are?''',

"Descriptive": '''**Descriptive Style**
   - Uses detailed descriptions with contextual information.
   - Example: I want to know the names and home addresses of all students older than 18.''',

"Concise": '''**Concise Style**
   - Use short sentences.
   - Example: Students older than 18, return their names and addresses.''',
"""

# Question Style Descriptions Dictionary (Base descriptions without examples)
STYLE_DESCRIPTIONS = {
    QuestionStyle.FORMAL: '''**Formal Style:**
- Uses standard grammar and professional vocabulary
- Complete, well-structured sentences with proper punctuation
- Neutral tone without personal references or emotional language
- No contractions, slang, or colloquialisms
- Clear and precise expression following formal writing conventions''',

    QuestionStyle.COLLOQUIAL: '''**Colloquial Style:**
- Employs informal vocabulary and conversational expressions
- Includes interjections ("Hey!", "Oh"), filler words, and casual phrases
- Uses contractions and relaxed grammar as in everyday speech
- May include personal pronouns (I, me, you) and emotional expressions
- Friendly, approachable tone with phrases like "I'd love to", "Could you help me"''',

    QuestionStyle.IMPERATIVE: '''**Imperative Style:**
- Uses command or directive sentences with urgency or necessity
- Includes polite request markers: "Could you please", "Please help me", "I need"
- Expresses urgency or importance: "I really need", "It's important", "Must have"
- May end with exclamation marks to convey emphasis
- Combines directive intent with polite or urgent phrasing''',

    QuestionStyle.INTERROGATIVE: '''**Interrogative Style:**
- Uses question forms that explicitly seek information
- Begins with phrases like: "Could you tell me", "Can you show me", "Would you explain"
- Or starts with interrogative words: "What", "Which", "Who", "Where", "How"
- Always ends with a question mark
- Polite inquiry tone without urgency or commands''',

    QuestionStyle.DESCRIPTIVE: '''**Descriptive Style:**
- States the information need directly: "I want to know", "I need to find out"
- May include brief context or purpose in the same sentence
- Declarative statements describing what information is desired
- More detailed than concise but not overly elaborate
- Focuses on describing the desired information rather than commanding or asking''',

    QuestionStyle.CONCISE: '''**Concise Style:**
- Uses short sentences with minimal words
- Includes essential verbs but eliminates unnecessary details
- Comma-separated phrases for brevity
- Direct and to-the-point without elaboration
- Maintains grammatical completeness while being brief'''
}

# Question Style Descriptions with Question Examples
STYLE_DESCRIPTIONS_WITH_QUESTION_EXAMPLES = {
    QuestionStyle.FORMAL: STYLE_DESCRIPTIONS[QuestionStyle.FORMAL] + '''

**Example Question:** "Find all students older than 18 years and return their home addresses."
**Distinguishing Features:** Standard grammar, neutral tone, no personal pronouns, formal vocabulary.''',

    QuestionStyle.COLLOQUIAL: STYLE_DESCRIPTIONS[QuestionStyle.COLLOQUIAL] + '''

**Example Question:** "Hey! Could you help me find all the students who are over 18? I'd love to know their names and where they live."
**Distinguishing Features:** Interjection "Hey!", "Could you help me", "I'd love to", conversational tone.''',

    QuestionStyle.IMPERATIVE: STYLE_DESCRIPTIONS[QuestionStyle.IMPERATIVE] + '''

**Example Question:** "Could you please gather all the students who are older than 18? I really need to know their names and where they live!"
**Distinguishing Features:** "Could you please", "I really need", exclamation mark showing urgency.''',

    QuestionStyle.INTERROGATIVE: STYLE_DESCRIPTIONS[QuestionStyle.INTERROGATIVE] + '''

**Example Question:** "Could you tell me which students are older than 18 and what their home addresses are?"
**Distinguishing Features:** "Could you tell me", question mark, inquiry-focused without urgency.''',

    QuestionStyle.DESCRIPTIVE: STYLE_DESCRIPTIONS[QuestionStyle.DESCRIPTIVE] + '''

**Example Question:** "I want to know the names and home addresses of all students older than 18."
**Distinguishing Features:** "I want to know", declarative statement, brief contextual purpose.''',

    QuestionStyle.CONCISE: STYLE_DESCRIPTIONS[QuestionStyle.CONCISE] + '''

**Example Question:** "Students older than 18, return their names and addresses."
**Distinguishing Features:** Short sentence, comma separation, includes verb "return", minimal words.'''
}

# Question Style Descriptions with Answer Examples
STYLE_DESCRIPTIONS_WITH_ANSWER_EXAMPLES = {
    QuestionStyle.FORMAL: STYLE_DESCRIPTIONS[QuestionStyle.FORMAL] + '''

**Example Answer:** "Provide all records from the engineering department pertaining to senior-level positions."''',

    QuestionStyle.COLLOQUIAL: STYLE_DESCRIPTIONS[QuestionStyle.COLLOQUIAL] + '''

**Example Answer:** "Hey! Could you help me get the engineering folks? I'd really love to see just the senior people there."''',

    QuestionStyle.IMPERATIVE: STYLE_DESCRIPTIONS[QuestionStyle.IMPERATIVE] + '''

**Example Answer:** "Please get me the engineering department records! I really need the senior positions right away."''',

    QuestionStyle.INTERROGATIVE: STYLE_DESCRIPTIONS[QuestionStyle.INTERROGATIVE] + '''

**Example Answer:** "Could you tell me which employees in the engineering department hold senior positions?"''',

    QuestionStyle.DESCRIPTIVE: STYLE_DESCRIPTIONS[QuestionStyle.DESCRIPTIVE] + '''

**Example Answer:** "I want to know which employees in the engineering department hold senior-level positions."''',

    QuestionStyle.CONCISE: STYLE_DESCRIPTIONS[QuestionStyle.CONCISE] + '''

**Example Answer:** "Engineering department, return senior positions."'''
}