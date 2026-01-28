import streamlit as st
import json
import random
from pathlib import Path
from dataset_dataclasses.question import Question, QuestionUnanswerable, QuestionStyle, QuestionDifficulty
from categories import get_category_by_name_and_subname
from db_datasets.db_dataset import DBDataset
from utils.style_and_difficulty_utils import DIFFICULTY_CRITERIA, STYLE_DESCRIPTIONS

# Configuration Constants
RANDOM_SEED = 42
NUM_QUESTIONS_TO_SAMPLE = 2

# File paths and database configuration
QUESTIONS_FILE_PATH = "example_results/dev_generated_question_v9_merged.json"
DB_ROOT_PATH = "../datasets/bird_dev/dev_databases"
DB_NAME = "BIRD"
OUTPUT_FILE_PATH = "annotated_questions.json"

# Quality factors to annotate (binary: correct/incorrect)
QUALITY_FACTORS = [
    "category",
    "style", 
    "difficulty",
    "sql_correct"
]


def load_questions(file_path: str) -> list[Question]:
    """Load questions from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    questions = []
    for item in data:        
        # Check if this is an unanswerable question
        if "hidden_knowledge" in item or "is_solvable" in item:
            question = QuestionUnanswerable.from_dict(item)
        else:
            question = Question.from_dict(item)
        questions.append(question)
    
    return questions


def sample_questions(questions: list[Question], seed: int, n: int) -> list[Question]:
    """Randomly sample n questions with given seed."""
    random.seed(seed)
    if len(questions) <= n:
        return questions
    return random.sample(questions, n)


def execute_sql_query(db_dataset: DBDataset, db_id: str, sql: str, limit: int = 5):
    """Execute SQL query and return results."""
    try:
        result = db_dataset.execute_query_with_columns(db_id, sql)
        if result is None:
            return None, "Query execution failed"
        
        columns, rows = result
        if not rows:
            return columns, []
        
        # Limit to first N rows
        limited_rows = rows[:limit]
        return columns, limited_rows
    except Exception as e:
        return None, f"Error: {str(e)}"


def get_database_schema_with_examples(db_dataset: DBDataset, db_id: str):
    """Get database schema with one example row per table."""
    try:
        import sqlite3
        import pandas as pd
        
        db_path = db_dataset._get_db_path(db_id)
        conn = sqlite3.connect(f'file:{db_path}?immutable=1', uri=True)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_info = {}
        
        for table in tables:
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table});")
            columns_info = cursor.fetchall()
            columns = [(col[1], col[2]) for col in columns_info]  # (name, type)
            
            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table});")
            foreign_keys_raw = cursor.fetchall()
            # Each row: (id, seq, table, from, to, on_update, on_delete, match)
            foreign_keys = []
            for fk in foreign_keys_raw:
                foreign_keys.append({
                    'from_column': fk[3],
                    'to_table': fk[2],
                    'to_column': fk[4]
                })
            
            # Get up to 3 example rows
            cursor.execute(f"SELECT * FROM {table} LIMIT 3;")
            example_rows = cursor.fetchall()
            
            schema_info[table] = {
                'columns': columns,
                'foreign_keys': foreign_keys,
                'example': example_rows
            }
        
        conn.close()
        return schema_info
    except Exception as e:
        return None


def initialize_session_state(questions: list[Question]):
    """Initialize session state variables."""
    if 'questions' not in st.session_state:
        st.session_state.questions = questions
    
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    if 'annotations' not in st.session_state:
        # Initialize annotations dict for each question
        st.session_state.annotations = {
            i: {factor: None for factor in QUALITY_FACTORS}
            for i in range(len(questions))
        }


def get_completion_status() -> tuple[int, int]:
    """Get number of completed annotations out of total."""
    completed = 0
    total = len(st.session_state.questions)
    
    for i in range(total):
        annotations = st.session_state.annotations[i]
        if all(v is not None for v in annotations.values()):
            completed += 1
    
    return completed, total


def save_annotations(output_path: str):
    """Save annotations to JSON file."""
    results = []
    
    for i, question in enumerate(st.session_state.questions):
        result = {
            "question": question.to_dict(),
            "quality_annotations": st.session_state.annotations[i]
        }
        results.append(result)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)


def main():
    st.set_page_config(page_title="Question Quality Annotation", layout="wide")
    st.title("Question Quality Annotation Tool")
    
    # Sidebar for configuration display
    with st.sidebar:
        st.header("Configuration")
        st.markdown(f"**Questions File:** `{QUESTIONS_FILE_PATH}`")
        st.markdown(f"**Output File:** `{OUTPUT_FILE_PATH}`")
        st.markdown(f"**DB Root Path:** `{DB_ROOT_PATH if DB_ROOT_PATH else 'Not configured'}`")
        st.markdown(f"**DB Name:** `{DB_NAME if DB_NAME else 'Not configured'}`")
        st.markdown("---")
        st.markdown(f"**Random Seed:** {RANDOM_SEED}")
        st.markdown(f"**Sample Size:** {NUM_QUESTIONS_TO_SAMPLE}")
    
    # Load questions button
    if st.sidebar.button("Load Questions") or 'questions' in st.session_state:
        if 'questions' not in st.session_state:
            try:
                questions = load_questions(QUESTIONS_FILE_PATH)
                sampled_questions = sample_questions(questions, RANDOM_SEED, NUM_QUESTIONS_TO_SAMPLE)
                initialize_session_state(sampled_questions)
                
                # Initialize DB dataset if paths provided
                if DB_ROOT_PATH and DB_NAME:
                    st.session_state.db_dataset = DBDataset(DB_ROOT_PATH, DB_NAME)
                
                st.sidebar.success(f"Loaded {len(sampled_questions)} questions")
            except Exception as e:
                st.sidebar.error(f"Error loading questions: {str(e)}")
                return
        
        # Display progress
        completed, total = get_completion_status()
        st.progress(completed / total if total > 0 else 0)
        st.markdown(f"**Progress:** {completed}/{total} questions annotated")
        
        # Get current question
        idx = st.session_state.current_index
        question: Question = st.session_state.questions[idx]
        
        # Display question number
        st.header(f"Question {idx + 1} of {len(st.session_state.questions)}")
        
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display question details
            st.subheader("Question Details")
            
            st.markdown(f"**Database ID:** {question.db_id}")
            
            st.markdown("**Question:**")
            st.info(question.question)
            
            if question.evidence:
                st.markdown("**Evidence:**")
                st.info(question.evidence)
            
            # Display Style and Difficulty
            st.markdown("---")
            st.markdown(f"**Style:** `{question.question_style.value}`")
            st.markdown(f"**Difficulty:** `{question.question_difficulty.value}`")
            
            # Display Category with details
            st.markdown("---")
            st.subheader("Category Information")
            st.markdown(f"**Name:** {question.category.get_name()}")
            if question.category.get_subname():
                st.markdown(f"**Subname:** {question.category.get_subname()}")
            
            # Determine category type
            if hasattr(question.category, 'is_answerable') and question.category.is_answerable():
                category_type = "answerable"
            elif hasattr(question.category, 'is_answerable') and not question.category.is_answerable():
                if hasattr(question.category, 'is_solvable') and question.category.is_solvable():
                    category_type = "ambiguous"
                else:
                    category_type = "unanswerable"
            else:
                category_type = "answerable"  # default
            st.markdown(f"**Category Type:** `{category_type}`")
            
            st.markdown("**Definition:**")
            st.info(question.category.get_definition())
            
            if hasattr(question.category, 'example') and question.category.get_examples():
                st.markdown("**Example:**")
                with st.expander("View Example"):
                    st.markdown(question.category.get_examples())
            
            # Display database schema with examples
            if hasattr(st.session_state, 'db_dataset'):
                st.markdown("---")
                with st.expander("View Database Schema & Examples"):
                    with st.spinner("Loading schema..."):
                        schema_info = get_database_schema_with_examples(
                            st.session_state.db_dataset,
                            question.db_id
                        )
                    
                    if schema_info:
                        for table_name, table_data in schema_info.items():
                            st.markdown(f"**Table: `{table_name}`**")
                            
                            # Display columns with types
                            columns_text = ", ".join([f"`{col[0]}` ({col[1]})" for col in table_data['columns']])
                            st.markdown(f"Columns: {columns_text}")
                            
                            # Display foreign keys
                            if table_data.get('foreign_keys'):
                                st.markdown("**Foreign Keys:**")
                                for fk in table_data['foreign_keys']:
                                    st.markdown(f"- `{fk['from_column']}` → `{fk['to_table']}.{fk['to_column']}`")
                            
                            # Display example rows
                            if table_data['example']:
                                import pandas as pd
                                column_names = [col[0] for col in table_data['columns']]
                                df = pd.DataFrame(table_data['example'], columns=column_names)
                                st.dataframe(df, width='stretch')
                            else:
                                st.markdown("*No data in this table*")
                            
                            st.markdown("---")
                    else:
                        st.error("Could not load database schema")
            
            # Display SQL and results OR feedback for non-SQL questions
            if question.sql:
                st.markdown("---")
                st.subheader("SQL Query")
                st.code(question.sql, language="sql")
                
                # Execute SQL if DB is configured
                if hasattr(st.session_state, 'db_dataset'):
                    with st.spinner("Executing query..."):
                        columns, result = execute_sql_query(
                            st.session_state.db_dataset,
                            question.db_id,
                            question.sql,
                            limit=5
                        )
                    
                    if columns is not None and isinstance(result, list):
                        st.markdown("**Query Results (up to 5 rows):**")
                        if result:
                            # Display as table
                            import pandas as pd
                            df = pd.DataFrame(result, columns=columns)
                            st.dataframe(df, width='stretch')
                        else:
                            st.warning("Query returned no results")
                    else:
                        st.error(f"Query execution error: {result}")
            else:
                # For questions without SQL, show feedback (hidden_knowledge)
                if isinstance(question, QuestionUnanswerable) and question.hidden_knowledge:
                    st.markdown("---")
                    st.subheader("Feedback / Hidden Knowledge")
                    st.info(question.hidden_knowledge)
                
                # Show database schema for non-SQL questions too
                if hasattr(st.session_state, 'db_dataset'):
                    st.markdown("---")
                    with st.expander("View Database Schema & Examples"):
                        with st.spinner("Loading schema..."):
                            schema_info = get_database_schema_with_examples(
                                st.session_state.db_dataset,
                                question.db_id
                            )
                        
                        if schema_info:
                            for table_name, table_data in schema_info.items():
                                st.markdown(f"**Table: `{table_name}`**")
                                
                                # Display columns with types
                                columns_text = ", ".join([f"`{col[0]}` ({col[1]})" for col in table_data['columns']])
                                st.markdown(f"Columns: {columns_text}")
                                
                                # Display foreign keys
                                if table_data.get('foreign_keys'):
                                    st.markdown("**Foreign Keys:**")
                                    for fk in table_data['foreign_keys']:
                                        st.markdown(f"- `{fk['from_column']}` → `{fk['to_table']}.{fk['to_column']}`")
                                
                                # Display example rows
                                if table_data['example']:
                                    import pandas as pd
                                    column_names = [col[0] for col in table_data['columns']]
                                    df = pd.DataFrame(table_data['example'], columns=column_names)
                                    st.dataframe(df, width='stretch')
                                else:
                                    st.markdown("*No data in this table*")
                                
                                st.markdown("---")
                        else:
                            st.error("Could not load database schema")
        
        with col2:
            # Quality annotations
            st.subheader("Quality Annotations")
            st.markdown("Rate each aspect as correct or incorrect:")
            
            annotations = st.session_state.annotations[idx]
            
            # Category quality
            st.markdown("**Category Correct?**")
            category_quality = st.radio(
                "Category matches the question?",
                options=[None, True, False],
                format_func=lambda x: "Not annotated" if x is None else ("✓ Correct" if x else "✗ Incorrect"),
                key=f"category_{idx}",
                index=0 if annotations["category"] is None else (1 if annotations["category"] else 2),
                label_visibility="collapsed"
            )
            st.session_state.annotations[idx]["category"] = category_quality
            
            st.markdown("---")
            
            # Style quality
            with st.expander("View Style Definition"):
                st.markdown(STYLE_DESCRIPTIONS[question.question_style])
            
            st.markdown("**Style Correct?**")
            style_quality = st.radio(
                "Style matches the question?",
                options=[None, True, False],
                format_func=lambda x: "Not annotated" if x is None else ("✓ Correct" if x else "✗ Incorrect"),
                key=f"style_{idx}",
                index=0 if annotations["style"] is None else (1 if annotations["style"] else 2),
                label_visibility="collapsed"
            )
            st.session_state.annotations[idx]["style"] = style_quality
            
            st.markdown("---")
            
            # Difficulty quality
            with st.expander("View Difficulty Criteria"):
                st.markdown(DIFFICULTY_CRITERIA[question.question_difficulty])
            
            st.markdown("**Difficulty Correct?**")
            difficulty_quality = st.radio(
                "Difficulty matches the question/SQL?",
                options=[None, True, False],
                format_func=lambda x: "Not annotated" if x is None else ("✓ Correct" if x else "✗ Incorrect"),
                key=f"difficulty_{idx}",
                index=0 if annotations["difficulty"] is None else (1 if annotations["difficulty"] else 2),
                label_visibility="collapsed"
            )
            st.session_state.annotations[idx]["difficulty"] = difficulty_quality
            
            st.markdown("---")
            
            # SQL/Feedback quality
            if question.sql:
                st.markdown("**SQL Correct?**")
                sql_quality = st.radio(
                    "SQL is semantically correct?",
                    options=[None, True, False],
                    format_func=lambda x: "Not annotated" if x is None else ("✓ Correct" if x else "✗ Incorrect"),
                    key=f"sql_{idx}",
                    index=0 if annotations["sql_correct"] is None else (1 if annotations["sql_correct"] else 2),
                    label_visibility="collapsed"
                )
            else:
                st.markdown("**Feedback Correct?**")
                sql_quality = st.radio(
                    "Feedback is actionable and clearly points at the issue written in the hidden knowledge?",
                    options=[None, True, False],
                    format_func=lambda x: "Not annotated" if x is None else ("✓ Correct" if x else "✗ Incorrect"),
                    key=f"sql_{idx}",
                    index=0 if annotations["sql_correct"] is None else (1 if annotations["sql_correct"] else 2),
                    label_visibility="collapsed"
                )
            st.session_state.annotations[idx]["sql_correct"] = sql_quality
        
        # Navigation buttons
        st.markdown("---")
        col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 1, 2])
        
        with col_nav1:
            if st.button("⬅ Back", disabled=(idx == 0)):
                st.session_state.current_index -= 1
                st.rerun()
        
        with col_nav2:
            if st.button("Next ➡", disabled=(idx == len(st.session_state.questions) - 1)):
                st.session_state.current_index += 1
                st.rerun()
        
        with col_nav3:
            # Recalculate completion status to check if finish button should be enabled
            completed_now, total_now = get_completion_status()
            
            # Show finish button if all questions are annotated
            if completed_now == total_now:
                if st.button("✓ Finish & Save", type="primary"):
                    try:
                        save_annotations(OUTPUT_FILE_PATH)
                        st.success(f"Annotations saved to {OUTPUT_FILE_PATH}")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error saving annotations: {str(e)}")
            else:
                st.button("✓ Finish & Save", disabled=True, help=f"Complete all {total_now} annotations to enable")
        
        with col_nav4:
            # Jump to question
            jump_to = st.number_input(
                "Jump to question:",
                min_value=1,
                max_value=len(st.session_state.questions),
                value=idx + 1,
                key="jump_to"
            )
            if st.button("Go"):
                st.session_state.current_index = jump_to - 1
                st.rerun()


if __name__ == "__main__":
    main()
