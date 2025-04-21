## Sygnals v1.5 - Enhanced Data Handling and Workflows

### 1. Introduction

This document outlines proposed features and enhancements for **Sygnals v1.5**. It assumes the successful implementation of the features described in the "Sygnals v1.0.0 Specification" document, including the refined data science workflow, core signal processing commands (`segment`, `features`, `augment`, `visualize`), and the advanced plugin system foundation.

Sygnals v1.5 significantly enhances data manipulation capabilities by introducing a dedicated **`sygnals data`** command group for powerful table- and database-style operations on tabular data. It also aims to improve usability and efficiency through advanced batch processing, an interactive mode, implemented dataset assembly, deeper plugin integration, and basic parallel processing support.

**Goals for v1.5:**

- Introduce a comprehensive `sygnals data` command group for rich querying, transformation, inspection, and export of tabular data.
- Introduce robust multi-step batch processing capabilities via workflow files.
- Provide an interactive mode (`REPL`) for exploratory analysis and step-by-step processing.
- Implement advanced dataset assembly methods (`vectors`, `sequences`, `image`) within `sygnals save dataset`.
- Enhance plugin capabilities for deeper integration, including custom data operations and I/O handlers.
- Introduce basic parallel processing support for performance gains.

### 2. Recap of v1.0.0 Foundation

Sygnals v1.5 builds upon the following key features established in v1.0.0:

- **Data Science Focused CLI:** Commands structured around common workflow stages (Load, Segment, Features, Augment, Save, Visualize).
- **Comprehensive Feature Framework:** `sygnals features extract` and `sygnals features transform scale`.
- **Advanced Plugin System:** Support for extending functionality via entry points or local plugins, managed by `sygnals plugin ...` and invoked via `--plugin` options within relevant commands (like `features`, `augment`, `visualize`).
- **Layered Configuration:** Robust configuration management via `sygnals.toml`, environment variables, and CLI arguments.
- **Core Signal Processing:** Base commands for segmentation, augmentation, feature extraction, filtering, and visualization of signal/audio data.

### 3. Proposed v1.5 Enhancements

#### 3.1. New Command Group: `sygnals data`

- **Motivation:** Provide a dedicated, powerful, and intuitive interface for manipulating tabular data (CSV, JSON, potentially SQL tables via plugins) directly within the Sygnals CLI, complementing the signal-oriented commands.

- **Proposed Feature:** A new top-level command group `sygnals data` with subcommands for inspection, querying, transformation, and export. This replaces the previously proposed standalone `query` and `transform` commands for tabular data workflows.

- Synopsis:

    Bash

    ```
    sygnals data [OPTIONS] COMMAND [ARGS]...
    ```

- **Subcommands:** `list`, `show`, `query`, `transform`, `export`, `plugin` (detailed in Section 4).

- **Data Flow:** Commands within this group are designed to be pipeable, operating on an implicit in-memory table (e.g., Pandas DataFrame) passed between steps. Input can be from a file or stdin.

#### 3.2. Advanced Batch Processing (`sygnals batch`)

- **Motivation:** Streamline complex, multi-step workflows involving multiple files and commands (including the new `data` commands).

- **Proposed Feature:** A `sygnals batch` command executing workflows defined in a configuration file (YAML/TOML).

- Synopsis:

    Bash

    ```
    sygnals batch --workflow <WORKFLOW_FILE> [--input-pattern <GLOB>] [--output-dir <DIR>] [--parallel <N>] [--params <JSON_OVERRIDES>]
    ```

- **Capabilities:** Define sequences of `sygnals` commands (including `data` subcommands and plugin commands), parameterize steps, define dependencies, override parameters, utilize parallel processing.

- Workflow Definition File (`workflow.yaml` Example):

    YAML

    ```
    name: Sensor Data Cleaning and Analysis
    description: Filters sensor logs, calculates derived values, exports summary.
    steps:
      - name: filter_errors
        command: data query # Using the new data command
        input: "{{ input_file }}" # Input file pattern handled by batch runner
        output: "{{ output_dir }}/filtered/{{ input_file.stem }}.filtered.csv"
        params:
          sql: "SELECT * FROM df WHERE status = 'OK' AND voltage > 2.5"
    
      - name: add_power_col
        command: data transform
        input: "{{ output_dir }}/filtered/{{ input_file.stem }}.filtered.csv"
        output: "{{ output_dir }}/transformed/{{ input_file.stem }}.power.csv"
        params:
          expr: "power_mW = voltage * voltage / 50 * 1000" # Example resistance 50 Ohm
        depends_on: filter_errors
    
      - name: export_summary_db
        command: data query
        input: "{{ output_dir }}/transformed/{{ input_file.stem }}.power.csv"
        # No direct file output, pipe to export
        persist: true # Keep result in memory for next step
        params:
          sql: "SELECT sensor_id, AVG(temperature_C) as avg_temp, AVG(power_mW) as avg_power FROM df GROUP BY sensor_id"
        depends_on: add_power_col
    
      - name: save_summary
        command: data export
        # Implicit input from previous persisted step
        output: "{{ output_dir }}/summary_{{ input_file.stem }}.db"
        params:
          format: sqlite
          table: sensor_summary
          mode: append # Append summaries from different files
        depends_on: export_summary_db
    ```

#### 3.3. Interactive Mode (`sygnals interactive`)

- **Motivation:** Facilitate exploratory analysis and step-by-step processing.

- **Proposed Feature:** An interactive REPL mode.

- **Synopsis:** `sygnals interactive`

- **Capabilities:** Enter `sygnals` subcommands directly, maintain state in variables, use variables in commands, introspection (`list`, `show`), tab completion, history.

- Example Session:

    ```
    sygnals interactive
    Sygnals v1.5 Interactive Mode. Type 'exit' or 'quit' to leave.
    >> # Use 'data' subcommands directly
    >> loaded_data = query sensor_log.csv --sql "SELECT timestamp, voltage FROM df WHERE status='OK'"
    Loaded DataFrame into variable 'loaded_data' (1500 rows, 2 columns)
    >> transformed_data = transform $loaded_data --expr "voltage_adj = voltage * 1.1"
    Transformed data stored in variable 'transformed_data' (1500 rows, 3 columns)
    >> show $transformed_data --head -n 3
    # Displays first 3 rows
    >> export $transformed_data -o adjusted_voltages.csv
    Data exported to adjusted_voltages.csv
    >> exit
    ```

#### 3.4. Enhanced Plugin Capabilities

- **Motivation:** Allow deeper plugin integration, especially for the new `data` commands and custom I/O.

- Proposed Features:

    - Data Plugin Hooks:

         Introduce specific hooks within 

        ```
        SygnalsPluginBase
        ```

         for the 

        ```
        data
        ```

         command group:

        - `register_data_expressions(self, registry)`: Register custom, named functions usable within `--expr` or a dedicated `--apply-plugin-expr` option in `data transform`.
        - `register_data_filters(self, registry)`: Register custom boolean filter functions usable via `--plugin-filter` in `data transform` or `data query`.
        - `register_data_exporters(self, registry)`: Register functions to handle custom export formats via `data export --format <CUSTOM>`.

    - **I/O Plugin Hooks:** Implement `register_data_loaders` and `register_data_savers` for handling custom file formats across relevant commands (e.g., `data query`, `data export`, potentially core `load`/`save`).

    - **Command Group Hook:** Implement `register_command_group` hook allowing plugins to add new top-level command groups.

    - **Entry Point:** Define a dedicated entry point group for data plugins: `sygnals.plugins.data`.

#### 3.5. Dataset Assembly Implementation

- **Motivation:** Fulfill the promise of the v1.0.0 spec.
- **Proposed Feature:** Implement core logic for `--assembly-method` options (`vectors`, `sequences`, `image`) in `sygnals save dataset`, allowing structured output formatting for ML tasks. Requires clear definition of input requirements (e.g., feature files, segment info) and relevant options (aggregation methods, padding, resizing).

#### 3.6. Parallel Processing

- **Motivation:** Improve performance for bulk operations.
- **Proposed Feature:** Add `--parallel <N>` option to `sygnals batch` (to parallelize file processing or independent steps) and potentially `sygnals features extract` (for parallel file processing).

### 4. Detailed Command Specifications (`sygnals data ...`)

This section details the subcommands for the new `sygnals data` group. These commands operate on tabular data loaded from files or piped from previous commands.

#### 4.1. `sygnals data list`

- **Purpose:** Quickly inspect columns and basic statistics of tabular data.

- **Synopsis:** `sygnals data list <INPUT>`

- **Input:** File path (CSV, JSON, etc.) or `-` for stdin.

- Options:

    - `-n, --num-rows INT`: Limit statistics calculation to the first N rows (default: 1000).
    - `--describe`: Show descriptive statistics (mean, std, min, max, quartiles) for numeric columns.
    - `--unique`: Show the count of unique values for each column.
    - `--non-null`: Show the count of non-null values for each column (default).
    - `--types`: Show the inferred data type for each column (default).

- **Output:** Formatted table to stdout summarizing column information.

- Example:

    Bash

    ```
    sygnals data list sensor_log.csv --describe --unique
    ```

#### 4.2. `sygnals data show`

- **Purpose:** Display sample rows (head, tail) or the entire dataset to the console.

- **Synopsis:** `sygnals data show <INPUT>`

- **Input:** File path or `-` for stdin.

- Options:

    - `-n, --num-rows INT`: Number of rows to display (default: 10).
    - `--head`: Show the first N rows (default).
    - `--tail`: Show the last N rows.
    - `--all`: Display all rows (use with caution).
    - `--format [tabulated|csv|json]`: Output format (default: tabulated).

- Example:

    Bash

    ```
    sygnals query sensor_log.csv --pandas "voltage < 2.0" | sygnals data show -n 5 --tail
    ```

#### 4.3. `sygnals data query`

- **Purpose:** Execute SQL queries or apply Pandas filters/selections on input data.

- **Synopsis:** `sygnals data query <INPUT> [QUERY_OPTIONS]`

- **Input:** File path or `-` for stdin.

- Query Options:

    - `--sql <SQL_QUERY>`: SQL query string (data accessible as `df`).
    - `--pandas <PANDAS_EXPR>`: Pandas boolean filter expression string.
    - `--select <COLUMNS>`: Comma-separated list of columns to select.
    - `--plugin-query <PLUGIN_NAME> [PLUGIN_ARGS...]`: Use a custom query plugin.
    - `--table <TEXT>`: Table name (primarily for SQL sources if supported via plugin).
    - `--engine [pandas|sqlite]`: Query engine for SQL (default: sqlite via pandasql).
    - `-p, --param KEY=VAL`: Bind parameters to SQL query (repeatable).
    - `--persist`: Keep the query result in memory for the *next* piped `sygnals data` command (e.g., before `export`).

- **Output:** Resulting data (DataFrame) piped to stdout (default CSV format for piping) or saved via `-o`.

- Example:

    Bash

    ```
    sygnals data query sensor_log.csv \
      --sql "SELECT sensor_id, AVG(temperature_C) AS avg_temp FROM df WHERE voltage > 3.0 GROUP BY sensor_id" \
      -o avg_temps.csv
    ```

#### 4.4. `sygnals data transform`

- **Purpose:** Apply transformations, filter, sort, rename, or drop columns. Operations are generally applied sequentially.

- **Synopsis:** `sygnals data transform <INPUT> [TRANSFORM_OPTIONS]`

- **Input:** File path or `-` for stdin.

- Transform Options:

    - `-e, --expr <PANDAS_EXPR>`: Pandas assignment expression (e.g., `'new_col = colA * 2'`). Repeatable.
    - `--filter <PANDAS_EXPR>`: Apply a Pandas boolean filter *after* any expressions.
    - `--drop-cols <COLUMNS>`: Comma-separated list of columns to drop.
    - `--rename <COL_MAP>`: Rename columns using format `old1:new1,old2:new2`.
    - `--sort-by <COLUMNS>`: Sort by columns (comma-separated). Prepend `-` for descending (e.g., `-timestamp,sensor_id`).
    - `--plugin <PLUGIN_NAME> [PLUGIN_ARGS...]`: Apply a custom transform plugin.
    - `--apply-plugin-expr <NAME> [PARAMS...]`: Apply a named expression registered by a plugin.

- **Output:** Transformed data (DataFrame) piped to stdout or saved via `-o`.

- Example:

    Bash

    ```
    sygnals data transform sensor_log.csv \
      --expr "temp_F = temperature_C * 9/5 + 32" \
      --filter "temp_F > 50" \
      --drop-cols "temperature_C,status" \
      --sort-by "-timestamp" \
      | sygnals data show -n 5
    ```

#### 4.5. `sygnals data export`

- **Purpose:** Save the data currently held in memory (usually piped from `query` or `transform`) to a specified destination and format.

- **Synopsis:** `sygnals data export <INPUT> [EXPORT_OPTIONS]`

- **Input:** File path or `-` for stdin (required).

- Export Options:

    - `-o, --output <PATH>`: **Required.** Destination path (file or potentially DB connection string via plugin). Extension often infers format.
    - `--format [csv|json|sqlite|sql|parquet|...]`: Explicitly set export format. Supports core types and custom formats via plugins.
    - `--table <TEXT>`: Table name required for database formats (sqlite, sql, plugin DBs).
    - `--mode [overwrite|append|fail]`: Behavior if the destination (table/file) exists (default: fail).
    - `--saver <PLUGIN_NAME> [PARAMS...]`: Use a custom saver plugin.

- Example:

    Bash

    ```
    # Query high voltage data and save to a new table in an SQLite DB
    sygnals data query sensor_log.csv --pandas "voltage > 4.5" \
      | sygnals data export - -o sensor_analysis.db --format sqlite --table high_voltage --mode overwrite
    ```

#### 4.6. `sygnals data plugin`

- **Purpose:** Manage data-specific plugins (list available data expressions, filters, exporters).
- **Synopsis:** `sygnals data plugin [list|...]`
- Subcommands:
    - `list [expressions|filters|exporters]`: List available extensions registered specifically for the `data` command group via the `sygnals.plugins.data` entry point.
- **Note:** Installation/enable/disable is still handled by the main `sygnals plugin` command group.

### 9. Plugin Extension Points (Data Specific)

- *(This section details hooks specific to the sygnals data command group, complementing the general plugin hooks in Section 3.3)*
- **Entry Point Group:** `sygnals.plugins.data` (for plugins primarily focused on tabular data manipulation).
- API Hooks (in `DataPluginBase` or `SygnalsPluginBase`):
    - `register_data_expressions(self, registry)`: Register named functions that take a DataFrame (or Series) and parameters, returning a modified DataFrame or Series. Usable via `--apply-plugin-expr`.
    - `register_data_filters(self, registry)`: Register named functions that take a DataFrame and parameters, returning a boolean Series for filtering. Usable via `--plugin-filter`.
    - `register_data_exporters(self, registry)`: Register functions that take a DataFrame and output path/parameters, handling saving to custom formats. Usable via `data export --format <CUSTOM_NAME>`.
    - *(Consider also `register_data_loaders`)*

### 10. Implementation Notes

- Leverage Pandas for in-memory data representation and transformations (`query`, `transform --expr`, `--filter`, etc.).
- Utilize `pandasql` or direct `sqlite3` in-memory databases for `--sql` queries.
- Implement an internal mechanism (e.g., passing DataFrames via temporary files or potentially in-memory objects if feasible within the CLI framework) to handle piping between `sygnals data` subcommands. The `--persist` flag in `query` explicitly signals the need to hold data for the next step.
- Reuse the core Sygnals configuration loading and the v1.0.0 plugin framework (`PluginLoader`, `PluginRegistry`) for discovering and managing `sygnals.plugins.data` plugins.
- Ensure robust error handling and clear CLI feedback for all data operations.
