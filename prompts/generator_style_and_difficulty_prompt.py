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


# Question Style Descriptions Dictionary
STYLE_DESCRIPTIONS = {
    QuestionStyle.FORMAL: '''**Formal Style:**
- Uses standard, professional grammar and vocabulary
- Maintains objective, business-like tone
- Avoids colloquialisms and emotional language
- Clear and precise phrasing

**Example:** "Identify all students who exceed 18 years of age and retrieve their residential addresses."''',

    QuestionStyle.COLLOQUIAL: '''**Colloquial Style:**
- Uses informal, conversational vocabulary
- May include casual expressions, interjections, or filler words
- Friendly and relaxed tone as if speaking to a colleague
- May use contractions and everyday language

**Example:** "Hey! Can you find all the students who are over 18? I'd love to know their names and where they live."''',

    QuestionStyle.IMPERATIVE: '''**Imperative Style:**
- Uses command or directive sentence structures
- Direct and action-oriented language
- Often includes words like "show," "list," "find," "get," "display"
- Clear instructions without question marks

**Example:** "Show me all students older than 18 with their home addresses."''',

    QuestionStyle.INTERROGATIVE: '''**Interrogative Style:**
- Poses questions using standard question forms
- Often begins with question words (who, what, where, when, why, how)
- Uses question marks and inquiry-based phrasing
- Polite and inquiry-focused

**Example:** "Which students are older than 18, and what are their home addresses?"''',

    QuestionStyle.DESCRIPTIVE: '''**Descriptive Style:**
- Provides context and background information
- Explains the purpose or motivation behind the request
- May include additional details about the use case
- Narrative approach to expressing information needs

**Example:** "I'm compiling a mailing list for a scholarship program. I need to identify students who are legally adults (over 18 years old) along with their current home addresses."''',

    QuestionStyle.CONCISE: '''**Concise Style:**
- Uses minimal words to convey the request
- Short, direct sentences or phrases
- Omits unnecessary details while maintaining clarity
- Efficient and to-the-point

**Example:** "Students over 18 with addresses."'''
}