# Database Tests

This directory contains comprehensive tests for the database functionality used in the AI bias research project.

## Structure

```
tests/database/
├── conftest.py              # Shared fixtures for all database tests
├── test_initialization.py   # Database creation and schema validation
├── test_insert_row.py      # Row insertion functionality
├── test_integration.py     # End-to-end workflow tests
└── README.md               # This file
```

## Test Categories

### Initialization Tests (`test_initialization.py`)
- Database file creation
- Table schema validation (columns, types, constraints)
- Handling existing databases

### Insert Row Tests (`test_insert_row.py`)
- Single and multiple row insertion
- Duplicate ID handling (`INSERT OR IGNORE`)
- Boolean `reasoning_model` field validation
- Debug output functionality
- Path handling (string vs Path objects)

### Integration Tests (`test_integration.py`)
- Complete workflow from initialization to querying
- Reasoning vs non-reasoning model differentiation
- Data integrity across operations
- Timestamp functionality

## Fixtures

### `temp_db`
Creates a temporary SQLite database for each test with automatic cleanup.

### `sample_data`
Provides test data matching the bias research schema:
- Standard models (GPT-4, Claude) with `reasoning_model=False`
- Reasoning models (o1-preview) with `reasoning_model=True` and token counts

## Running Tests

```bash
# Run all database tests
pytest tests/database/ -v

# Run specific test categories
pytest tests/database/test_initialization.py -v
pytest tests/database/test_insert_row.py -v
pytest tests/database/test_integration.py -v

# Run with coverage
pytest tests/database/ --cov=bias_in_llms.utils.database
```

## Database Schema

The tests validate this schema:

```sql
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY,
    full_prompt TEXT,
    question TEXT,
    option_A TEXT,
    option_B TEXT,
    answer TEXT,
    model_id TEXT,
    reasoning_model BOOLEAN,
    reasoning_tokens INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Key Features Tested

- ✅ Database initialization and table creation
- ✅ Row insertion with duplicate prevention
- ✅ Boolean field handling for reasoning models
- ✅ Automatic timestamp generation
- ✅ Path flexibility (string/Path objects)
- ✅ Data integrity and consistency
- ✅ Debug output functionality 