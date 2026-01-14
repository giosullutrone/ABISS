import sqlite3
import time
from typing import Any
import json
import dataclasses
import os
from json import JSONEncoder
import re


# From: https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/gpt_request.py#L44
def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]

    # Print the column names
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    # print(header)
    # Print the values
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output

def get_tables_and_cols(db_path, db_sql_manipulation: str | None=None) -> dict[str, list[str]]:
    """
    Get the column names of the first table in the database.
    If db_sql_manipulation is provided, it will be executed before fetching the column names.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # If db_sql_manipulation is provided, manipulate the schema first
    if db_sql_manipulation:
        cursor.execute(db_sql_manipulation)
    
    columns = {}
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas: dict[str, list[str]] = {}
    for table in tables:
        if table == 'sqlite_sequence':
            continue
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt

        column_names = [description[0] for description in cursor.description]
        if column_names:
            # If the table has columns, add them to the list
            columns[table[0]] = column_names
        else:
            # If the table has no columns, add an empty list
            columns[table[0]] = []
    conn.close()
    # Convert the dictionary to a list of tuples
    return schemas

# From: https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/gpt_request.py#L60
def generate_schema_prompt(db_path, num_rows=None, db_sql_manipulation: str | None=None):
    # extract create ddls
    '''
    :param root_place:
    :param db_name:
    :return:
    '''
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()

    ####################
    # Ours
    ####################
    # if db_sql_manipulation is provided, manipulate the schema first
    if db_sql_manipulation:
        cursor.execute(db_sql_manipulation)
    ####################

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table == 'sqlite_sequence':
            continue
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        if num_rows:
            cur_table = table[0]
            if cur_table in ['order', 'by', 'group']:
                cur_table = "`{}`".format(cur_table)

            cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(num_rows, cur_table, num_rows, rows_prompt)
            schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt

def get_table_columns(db_path: str, table_name: str, db_sql_manipulation: str | None=None) -> list[str] | None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # If db_sql_manipulation is provided, manipulate the schema first
    if db_sql_manipulation:
        cursor.execute(db_sql_manipulation)

    cursor.execute(f"PRAGMA table_info({table_name});")
    columns_info = cursor.fetchall()
    column_names = [info[1] for info in columns_info]  # The second element is the column name
    conn.close()

    if not column_names:
        return None
    return column_names

def get_table_id_columns(db_path: str, table_name: str, db_sql_manipulation: str | None=None) -> list[str] | None:
    # Get the primary key columns for the table in the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # If db_sql_manipulation is provided, manipulate the schema first
    if db_sql_manipulation:
        cursor.execute(db_sql_manipulation)

    cursor.execute(f"PRAGMA table_info({table_name});")
    columns_info = cursor.fetchall()
    id_columns = [info[1] for info in columns_info if info[5] == 1]  # The sixth element indicates if it's a primary key
    conn.close()
    if not id_columns:
        return None
    return id_columns


# From: https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/gpt_request.py#L113
# IMPORTANT: A small change to the original code, to add ```sql``` blocks in the prompt
def cot_wizard():
    cot = "\nUse ```sql``` blocks to delimit the SQL.\nGenerate the SQL after thinking step by step: "
    return cot

# From: https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/gpt_request.py#L99
def generate_comment_prompt(question, knowledge=None):
    pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_prompt_kg = "-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above."
    # question_prompt = "-- {}".format(question) + '\n SELECT '
    question_prompt = "-- {}".format(question)
    knowledge_prompt = "-- External Knowledge: {}".format(knowledge)

    if not knowledge_prompt:
        result_prompt = pattern_prompt_no_kg + '\n' + question_prompt
    else:
        result_prompt = knowledge_prompt + '\n' + pattern_prompt_kg + '\n' + question_prompt

    return result_prompt

# From: https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/gpt_request.py#L138
# Modified to accept knowledge and db_sql_manipulation
def generate_combined_prompts_one(db_path, question, knowledge=None, num_rows=None, db_sql_manipulation: str | None=None):
    schema_prompt = generate_schema_prompt(db_path, num_rows=num_rows, db_sql_manipulation=db_sql_manipulation) # This is the entry to collect values
    comment_prompt = generate_comment_prompt(question, knowledge)

    combined_prompts = schema_prompt + '\n\n' + comment_prompt + cot_wizard() + '\nSELECT '
    return combined_prompts

def execute_sql_query(
    db_path: str,
    sql_query: str,
    db_sql_manipulation: str | None = None,
    max_seconds: float | None = 30.0,
) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Set up timeout via progress handler
    if max_seconds is not None:
        start = time.monotonic()

        def progress_handler() -> int:
            # Called every N VM steps; return non-zero to abort
            if time.monotonic() - start > max_seconds:
                return 1  # abort query -> raises sqlite3.OperationalError
            return 0

        # Call handler every 1000 "virtual machine" steps (tune as needed)
        conn.set_progress_handler(progress_handler, 1000)

    try:
        if db_sql_manipulation:
            cursor.execute(db_sql_manipulation)
        cursor.execute(sql_query)
        results = cursor.fetchall()
    finally:
        # Always clear handler and close connection
        conn.set_progress_handler(None, 0)
        conn.close()

    return results

class EnhancedJSONEncoder(JSONEncoder):
    def default(self, o):
        # Check if the object is a dataclass instance
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o) # type: ignore[return-value]
        # Check if the object is a class (not an instance)
        if isinstance(o, type):
            return o.__name__
        try:
            # Attempt to use the default serialization
            return super().default(o)
        except TypeError:
            # If serialization fails, return the class name of the instance
            return o.__class__.__name__

def save_dataclass(path: str, d) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=4, cls=EnhancedJSONEncoder)

def extract_last_sql_query_from_block(text: str) -> str | None:
    """
    Extract the last ```sql ... ``` code block from text.
    """
    # Match ```sql … ```  (the “sql” tag is optional, case-insensitive)
    pattern = re.compile(
        r"```(?:\s*sql)?\s*(.*?)```",     # capture everything between the fences
        re.IGNORECASE | re.DOTALL
    )

    blocks = pattern.findall(text)
    if not blocks:               # No SQL fenced blocks present
        return None

    return blocks[-1].strip()    # Return the last one