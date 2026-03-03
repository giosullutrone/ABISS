# Database Setup Guide

This guide explains how to organize databases for use with the ABISS generation and interaction pipelines.

## Required Folder Structure

ABISS expects databases organized in a BIRD-style directory layout. Each database lives in its own folder under a shared root directory:

```
databases/
├── my_database/
│   ├── my_database.sqlite
│   └── database_description/    # optional, follows BIRD convention
│       ├── table_one.csv
│       ├── table_two.csv
│       └── ...
├── another_database/
│   ├── another_database.sqlite
│   └── database_description/
│       ├── employees.csv
│       └── departments.csv
└── ...
```

**Key conventions:**
- The SQLite file must be named identically to its parent folder (for instance, `california_schools/california_schools.sqlite`)
- The pipeline reads schema information directly from the SQLite database (DDL statements and sample rows). The `database_description/` subdirectory is a BIRD convention for column-level documentation and is not read by the current code, but is recommended for dataset reproducibility.

## SQLite Database File

Each database must be a standard SQLite file (`.sqlite` extension). The system opens connections in immutable mode (`?immutable=1`) and enforces a default timeout of 3 seconds per query.

## Schema Documentation (database_description/) (Optional)

The BIRD dataset includes a `database_description/` folder with one CSV file per table, documenting column semantics. While the current ABISS pipeline reads schema information directly from the SQLite file itself, keeping these CSV files alongside your database follows the standard BIRD convention and supports dataset reproducibility.

Each CSV uses five fields:

```
original_column_name,column_name,column_description,data_format,value_description
```

| Field | Description |
|-------|-------------|
| `original_column_name` | Column name as stored in SQLite |
| `column_name` | User-friendly display name (can be empty) |
| `column_description` | Natural language description of the column |
| `data_format` | Data type: `integer`, `text`, `real`, `date`, etc. |
| `value_description` | Additional details about possible values, encodings, or domain knowledge |

**Example** (`employees.csv`):
```csv
employee_id,,unique identifier for each employee,integer,
name,,full name of the employee,text,
department,,department the employee belongs to,text,"HR, Engineering, Sales, Marketing"
hire_date,,date the employee was hired,date,format: YYYY-MM-DD
salary,,annual salary in USD,real,
```

## Adding a New Database

Follow these steps to add your own database:

1. **Create the database folder** under your chosen root directory:
   ```bash
   mkdir -p datasets/my_databases/my_new_db
   ```

2. **Place your SQLite file** with the matching name:
   ```bash
   cp /path/to/your/database.sqlite datasets/my_databases/my_new_db/my_new_db.sqlite
   ```

3. **(Optional) Create schema documentation** following the BIRD convention:
   ```bash
   mkdir -p datasets/my_databases/my_new_db/database_description
   # For each table in your database, create a CSV:
   # datasets/my_databases/my_new_db/database_description/table_name.csv
   ```

4. **Verify** the database is discoverable:
   ```bash
   ls datasets/my_databases/my_new_db/
   # Expected: my_new_db.sqlite
   ```

## Using Your Database in the Pipelines

### Question Generation

```bash
python do_question_generation.py \
    --db_name my_databases \
    --db_root_path datasets/my_databases \
    --db_ids my_new_db \
    --model_names models/Qwen2.5-32B-Instruct \
    --n_samples 5 \
    --output_path results/my_new_db_questions.json \
    --verbose
```

- `--db_root_path`: the root directory containing all database folders
- `--db_ids`: select specific databases (omit to use all databases under the root)

### Interactive Benchmarking

```bash
python do_interaction.py \
    --db_name my_databases \
    --db_root_path datasets/my_databases \
    --db_ids my_new_db \
    --question_path results/my_new_db_questions.json \
    --model_names models/Qwen2.5-32B-Instruct \
    --output_path results/my_new_db_interactions.json \
    --verbose
```

## Notes

- All databases in the root directory are automatically discovered by the pipeline. Use `--db_ids` to restrict to specific ones.
- The pipeline extracts schema information (DDL statements and sample rows) directly from the SQLite file. The `database_description/` CSVs are not read by the code but are recommended for documenting your dataset following the BIRD convention.
- The system supports any standard SQLite database. There are no constraints on the number of tables, columns, or rows.
