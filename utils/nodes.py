"""Node definitions for the analytical chatbot."""

import re
import yaml
from pocketflow import Node
from utils import call_llm, execute_sandboxed_code, parse_code_block

# Try to import database utilities
try:
    from utils.database import get_schema_description, HAS_DUCKDB
except ImportError:
    HAS_DUCKDB = False
    def get_schema_description():
        return "Database not available"


class GetInputNode(Node):
    """Receives user message and initializes the processing context."""

    def prep(self, shared):
        # Input comes from web request, already in shared
        return shared.get("user_message", "")

    def exec(self, user_message):
        # No processing needed, just pass through
        return user_message

    def post(self, shared, prep_res, exec_res):
        # Ensure user_message is set
        if not exec_res:
            shared["response"] = {
                "message": "I didn't receive a message. How can I help you?",
                "code": None,
                "output": None,
                "plots": None,
                "tables": None,
                "html": None,
                "error": None
            }
            return "output"  # Skip to output
        return "default"


class ClassifyIntentNode(Node):
    """Classifies user intent: conversation or code_execution."""

    def prep(self, shared):
        user_message = shared.get("user_message", "")
        chat_history = shared.get("chat_history", [])
        uploaded_files = shared.get("uploaded_files", {})

        # Get file info for context
        file_info = []
        for fname, df in uploaded_files.items():
            if hasattr(df, 'columns'):
                file_info.append(f"- {fname}: columns={list(df.columns)}, rows={len(df)}")
            else:
                file_info.append(f"- {fname}: data available")

        return {
            "message": user_message,
            "history": chat_history[-6:],  # Last 3 exchanges
            "files": file_info
        }

    def exec(self, prep_res):
        files_desc = "\n".join(prep_res["files"]) if prep_res["files"] else "None"

        # Format history for context
        history_text = ""
        for msg in prep_res["history"]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # Truncate
            history_text += f"{role}: {content}\n"

        prompt = f"""Classify the user's intent based on their message.

Available data files:
{files_desc}

Database available: Yes (tables: employees, products, sales, customers)

Recent conversation:
{history_text}

Current user message: {prep_res["message"]}

Determine if the user wants:
- "conversation": General chat, greetings, questions about the bot, clarifications,
  asking what you can do, etc.
- "code_execution": Data analysis, calculations, statistics, visualizations,
  working with uploaded data, mathematical operations, generating charts/plots,
  INTERACTIVE DASHBOARDS, HTML visualization,
  DATABASE QUERIES, SQL operations, querying tables, summarizing data,
  showing table contents, any request involving the database tables.

IMPORTANT: Any request to query, summarize, describe contents, show data, or analyze
the database tables (employees, products, sales, customers) should be "code_execution".

Respond in YAML format:
```yaml
intent: conversation | code_execution
reason: brief explanation
```"""

        response = call_llm(prompt)

        # Parse YAML response
        yaml_str = response.split("```yaml")[1].split("```")[0].strip() \
            if "```yaml" in response else response
        result = yaml.safe_load(yaml_str)

        if not isinstance(result, dict):
            # If the LLM refused or returned plain text, raise error to trigger retry
            raise ValueError(f"Invalid output format (expected YAML dict, got {type(result)}): {response}")

        # Handle case where keys are missing or intent is invalid
        if "intent" not in result:
             raise ValueError(f"Missing 'intent' key in response: {response}")
             
        if result["intent"] not in ["conversation", "code_execution"]:
             raise ValueError(f"Invalid intent '{result.get('intent')}' in response: {response}")

        return result

    def post(self, shared, prep_res, exec_res):
        shared["intent"] = exec_res["intent"]
        return exec_res["intent"]


class ConversationResponseNode(Node):
    """Generates a conversational response for non-code queries."""

    def prep(self, shared):
        return {
            "message": shared.get("user_message", ""),
            "history": shared.get("chat_history", [])[-6:],
            "files": list(shared.get("uploaded_files", {}).keys())
        }

    def exec(self, prep_res):
        files_list = ", ".join(prep_res["files"]) if prep_res["files"] else "no files uploaded yet"

        # Get database info
        db_info = ""
        if HAS_DUCKDB:
            db_info = """
- Query the built-in DuckDB database with SQL (tables: employees, products, sales, customers)"""

        system_prompt = f"""You are a helpful analytical assistant. You can:
- Have friendly conversations
- Analyze data when users upload CSV/JSON files
- Generate Python code for calculations and visualizations
- Create charts and plots using Altair{db_info}
- Create diagrams using Mermaid.js syntax (wrap in ```mermaid code blocks)

Currently uploaded files: {files_list}
Database: {"Available (employees, products, sales, customers tables)" if HAS_DUCKDB else "Not available"}

Be helpful, concise, and friendly. If the user asks what you can do, explain your
analytical capabilities including the database tables."""

        # Build messages from history
        messages = []
        for msg in prep_res["history"]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        response = call_llm(
            prompt=prep_res["message"],
            system_prompt=system_prompt,
            messages=messages
        )

        return response

    def post(self, shared, prep_res, exec_res):
        shared["response"] = {
            "message": exec_res,
            "code": None,
            "output": None,
            "plots": None,
            "tables": None,
            "html": None,
            "error": None
        }
        return "output"


class GenerateCodeNode(Node):
    """Generates Python code for analytical tasks."""

    def prep(self, shared):
        user_message = shared.get("user_message", "")
        uploaded_files = shared.get("uploaded_files", {})
        chat_history = shared.get("chat_history", [])[-6:]

        # Build detailed file descriptions
        file_descriptions = []
        file_vars = {}

        for fname, df in uploaded_files.items():
            # Calculate variable name exactly as ExecuteCodeNode does
            var_name = fname.rsplit('.', 1)[0].replace(' ', '_').replace('-', '_')
            file_vars[fname] = var_name

            # Support both Polars and Pandas DataFrames
            if hasattr(df, 'schema'):  # Polars DataFrame
                desc = f"""File: {fname} (Available as Polars DataFrame variable: {var_name})
  Columns: {df.columns}
  Schema: {df.schema}
  Shape: ({df.height}, {df.width})
  Sample values: {df.head(2).to_dicts()}"""
                file_descriptions.append(desc)
            elif hasattr(df, 'columns') and hasattr(df, 'dtypes'):  # Pandas DataFrame
                desc = f"""File: {fname} (Available as Pandas DataFrame variable: {var_name})
  Columns: {list(df.columns)}
  Types: {df.dtypes.to_dict()}
  Shape: {df.shape}
  Sample values: {df.head(2).to_dict()}"""
                file_descriptions.append(desc)

        return {
            "message": user_message,
            "files": file_descriptions,
            "file_vars": file_vars,
            "history": chat_history
        }

    def exec(self, prep_res):
        files_desc = "\n\n".join(prep_res["files"]) if prep_res["files"] else "No files uploaded"

        # Construct explicit variable list string
        var_list_str = ", ".join([f"{v} (for {k})" for k, v in prep_res["file_vars"].items()])

        # Get database schema if available
        db_schema = get_schema_description() if HAS_DUCKDB else "Database not available"

        system_prompt = """You are a Python code generator for data analysis. Generate clean,
executable Python code that:
- PREFER Polars (as pl) over Pandas for data manipulation - it's faster and more expressive
- Use numpy (as np) for numerical operations when needed
- Use json for JSON serialization when needed
- Use Altair (as alt) for ALL visualizations - do NOT use matplotlib
- Use query_db(sql) to query the DuckDB database - returns a Polars DataFrame
- Use show_table(df) to display DataFrames as nicely formatted tables
- Use show_html(content) to render HTML content (e.g. interactive dashboards)

CRITICAL RULES:
- DataFrames are already loaded as POLARS DataFrames. Use the EXACT variable names provided.
- Use show_table(df) or show_table(df, "Title") to display DataFrames - this renders them as nice HTML tables
- Use print() for simple text output (numbers, strings, summaries)
- For ALL charts: assign to a variable named 'chart' - the system will render it automatically
- NEVER call .show(), .display(), or .save() on charts - just assign to 'chart' variable
- Altair works with Polars DataFrames directly - no conversion needed. Use alt.Chart(df)...
- For INTERACTIVE DASHBOARDS: generate the HTML string (e.g. using altair chart.save(None, format='html') or constructing it) and pass it to show_html(html_str).
- RESPONSIVENESS: When creating dashboards, ensure they resize with the window. Use `width='container'` in Altair charts and CSS `width: 100%` for HTML containers.

DISPLAYING DATA:
- show_table(df) - Display a DataFrame as a nicely formatted HTML table
- show_table(df, "Title") - Display with a title header
- show_html(str) - Display raw HTML content (useful for dashboards)
- print(df.to_markdown()) - Use this if the user asks for a text report/markdown table to be embedded in the final answer
- print() - Use for simple values, summaries, or text output
- Example:
    df = query_db("SELECT * FROM employees")
    show_table(df, "All Employees")  # Shows as formatted table

DATABASE QUERIES:
- Use query_db("SELECT ...") to query the database
- Results are returned as Polars DataFrames
- Example: df = query_db("SELECT name, salary FROM employees WHERE salary > 80000")
- You can combine database queries with uploaded file data

SAVING & LOADING INTERMEDIATE RESULTS:
- save_table(df, "name") - Save a DataFrame to DuckDB as 'saved_name' table
- load_table("name") - Load a previously saved table as Polars DataFrame
- list_saved_tables() - Show all saved tables
- delete_table("name") - Delete a saved table
- Use these for multi-step analysis workflows where you need to persist intermediate results

POLARS SYNTAX REMINDERS:
- Select columns: df.select(['col1', 'col2']) or df.select(pl.col('col1'))
- Filter rows: df.filter(pl.col('age') > 30)
- Group by: df.group_by('category').agg(pl.col('value').mean())
- Sort: df.sort('column', descending=True)
- Add column: df.with_columns((pl.col('a') + pl.col('b')).alias('sum'))
- Descriptive stats: df.describe()
- Value counts: df['column'].value_counts()
- Null handling: df.drop_nulls() or pl.col('x').fill_null(0)

ALTAIR EXAMPLES:
  chart = alt.Chart(df).mark_bar().encode(x='category:N', y='count()')
  chart = alt.Chart(df).mark_bar().encode(alt.X('age:Q', bin=True), y='count()')
  chart = alt.Chart(df).mark_line().encode(x='date:T', y='value:Q')
  chart = alt.Chart(df).mark_point().encode(x='x:Q', y='y:Q', color='category:N')"""

        prompt = f"""Generate Python code to answer this request: {prep_res["message"]}

Available DataFrames (pre-loaded as Polars): {var_list_str if var_list_str else 'None'}

Uploaded File Schema:
{files_desc}

{db_schema}

Generate only the Python code, wrapped in ```python``` blocks. Include comments explaining
each step. Use Polars syntax for data manipulation. Use query_db() for database queries."""

        response = call_llm(prompt=prompt, system_prompt=system_prompt)
        code = parse_code_block(response)

        # Remove any .show(), .display(), .save() calls that LLM might add
        code = re.sub(r'\.show\s*\([^)]*\)', '', code)
        code = re.sub(r'\.display\s*\([^)]*\)', '', code)
        code = re.sub(r'\.save\s*\([^)]*\)', '', code)
        code = re.sub(r'\bplt\.show\s*\([^)]*\)', '', code)

        return code

    def post(self, shared, prep_res, exec_res):
        shared["generated_code"] = exec_res
        return "default"


class ExecuteCodeNode(Node):
    """Executes generated code in a sandboxed environment."""

    def prep(self, shared):
        code = shared.get("generated_code", "")
        uploaded_files = shared.get("uploaded_files", {})

        # Prepare context with uploaded data
        context = {}
        for fname, df in uploaded_files.items():
            # Create variable name from filename (remove extension, replace spaces)
            var_name = fname.rsplit('.', 1)[0].replace(' ', '_').replace('-', '_')
            context[var_name] = df

        return {"code": code, "context": context}

    def exec(self, prep_res):
        result = execute_sandboxed_code(
            code=prep_res["code"],
            context=prep_res["context"],
            timeout_seconds=30
        )
        return result

    def post(self, shared, prep_res, exec_res):
        shared["execution_result"] = exec_res
        return "default"


class FormatResultsNode(Node):
    """Formats execution results for user display."""

    def prep(self, shared):
        return {
            "user_message": shared.get("user_message", ""),
            "code": shared.get("generated_code", ""),
            "result": shared.get("execution_result", {})
        }

    def exec(self, prep_res):
        result = prep_res["result"]
        code = prep_res["code"]

        if result.get("success"):
            output = result.get("stdout", "").strip()
            if output:
                # Generate a brief explanation
                prompt = f"""The user asked: {prep_res["user_message"]}

The following Python code was executed:
```python
{code}
```

And produced this output:
{output}

Respond to the user's request using the code execution results.
- If the user asked for a "report" or detailed analysis, provide a comprehensive answer, including relevant data tables (formatted as Markdown) and analysis.
- IMPORTANT: Do NOT wrap Markdown tables in code blocks (```). Output them directly so they render as tables.
- Otherwise, provide a brief (1-2 sentence) natural language summary of the results.
- Be specific about the numbers/findings.
- You can also use Mermaid diagrams (```mermaid) if helpful."""

                explanation = call_llm(prompt)
            else:
                explanation = "The code executed successfully."

            return {
                "message": explanation,
                "code": code,
                "output": output if output else "Code executed successfully (no output)",
                "plots": result.get("plots", []),
                "tables": result.get("tables", []),
                "html": result.get("html", []),
                "error": None
            }
        else:
            error = result.get("error", "Unknown error")
            return {
                "message": f"I encountered an error while running the code: {error}",
                "code": code,
                "output": None,
                "plots": None,
                "tables": None,
                "html": None,
                "error": error
            }

    def post(self, shared, prep_res, exec_res):
        shared["response"] = exec_res
        return "default"


class OutputResponseNode(Node):
    """Finalizes response and updates chat history."""

    def prep(self, shared):
        return {
            "user_message": shared.get("user_message", ""),
            "response": shared.get("response", {})
        }

    def exec(self, prep_res):
        # No additional processing needed
        return prep_res

    def post(self, shared, prep_res, exec_res):
        # Update chat history
        if "chat_history" not in shared:
            shared["chat_history"] = []

        shared["chat_history"].append({
            "role": "user",
            "content": exec_res["user_message"]
        })

        # Build assistant message content
        response = exec_res["response"]
        assistant_content = response.get("message", "")
        if response.get("code"):
            assistant_content += f"\n\n```python\n{response['code']}\n```"
        if response.get("output"):
            assistant_content += f"\n\nOutput:\n```\n{response['output']}\n```"

        shared["chat_history"].append({
            "role": "assistant",
            "content": assistant_content
        })

        return None  # End of flow
