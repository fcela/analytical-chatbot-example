"Node definitions for the analytical chatbot."

import re
import yaml
from pocketflow import Node
from utils import call_llm, parse_code_block

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
        return shared.get("user_message", "")

    def exec(self, user_message):
        return user_message

    def post(self, shared, prep_res, exec_res):
        if not exec_res:
            shared["response"] = {
                "message": "I didn't receive a message. How can I help you?",
                "code": None,
                "output": None,
                "artifacts": {},
                "error": None
            }
            return "output"
        return "default"


class ClassifyIntentNode(Node):
    """Classifies user intent: conversation or code_execution."""

    def prep(self, shared):
        user_message = shared.get("user_message", "")
        chat_history = shared.get("chat_history", [])
        uploaded_files = shared.get("uploaded_files", {}) # Now just metadata

        file_info = []
        for fname, meta in uploaded_files.items():
            # meta might be {"size": 1234}
            file_info.append(f"- {fname}")

        return {
            "message": user_message,
            "history": chat_history[-6:],
            "files": file_info
        }

    def exec(self, prep_res):
        files_desc = "\n".join(prep_res["files"]) if prep_res["files"] else "None"

        history_text = ""
        for msg in prep_res["history"]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            history_text += f"{role}: {content}\n"

        prompt = f"""Classify the user's intent based on their message.

Available data files:
{files_desc}

Database available: Yes (tables: employees, products, sales, customers)

Recent conversation:
{history_text}

Current user message: {prep_res["message"]}

Determine if the user wants:
- \"conversation\": General chat, greetings, questions about the bot.
- \"code_execution\": Data analysis, calculations, statistics, visualizations,
  working with uploaded data, generating charts/plots, database queries.

IMPORTANT: Any request to query, summarize, describe contents, show data, or analyze
the database tables should be \"code_execution\".

Respond in YAML format:
```yaml
intent: conversation | code_execution
reason: brief explanation
```"""

        response = call_llm(prompt)
        yaml_str = response.split("```yaml")[1].split("```")[0].strip() \
            if "```yaml" in response else response
        result = yaml.safe_load(yaml_str)

        if not isinstance(result, dict) or "intent" not in result:
             raise ValueError(f"Invalid response: {response}")
             
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
        db_info = "\n- Query the built-in DuckDB database" if HAS_DUCKDB else ""

        system_prompt = f"""You are a helpful analytical assistant. You can:
- Have friendly conversations
- Analyze data when users upload CSV/JSON files
- Generate Python code for calculations and visualizations
- Create charts and plots using Altair{db_info}

Currently uploaded files: {files_list}

Be helpful, concise, and friendly."""

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
            "artifacts": {},
            "error": None
        }
        return "output"


class GenerateCodeNode(Node):
    """Generates Python code for analytical tasks."""

    def prep(self, shared):
        user_message = shared.get("user_message", "")
        uploaded_files = shared.get("uploaded_files", {}) # Metadata
        chat_history = shared.get("chat_history", [])[-6:]

        file_descriptions = []
        file_vars = {}

        for fname in uploaded_files.keys():
            var_name = fname.rsplit('.', 1)[0].replace(' ', '_').replace('-', '_')
            file_vars[fname] = var_name
            # Since we don't have the DF here (it's in the kernel), we just describe it generically
            # Ideally, the kernel could return schema metadata upon loading.
            # For now, we tell LLM it exists.
            desc = f"File: {fname} (Loaded as Polars DataFrame variable: {var_name})"
            file_descriptions.append(desc)

        return {
            "message": user_message,
            "files": file_descriptions,
            "file_vars": file_vars,
            "history": chat_history
        }

    def exec(self, prep_res):
        files_desc = "\n".join(prep_res["files"]) if prep_res["files"] else "No files uploaded"
        var_list_str = ", ".join([f"{v} (for {k})" for k, v in prep_res["file_vars"].items()])
        db_schema = get_schema_description() if HAS_DUCKDB else "Database not available"

        system_prompt = """You are a Python code generator for data analysis. Generate clean,
executable Python code that:
- PREFER Polars (as pl) over Pandas for data manipulation
- Use Altair (as alt) for ALL visualizations
- Use query_db(sql) to query the DuckDB database
- Use show_table(df) to display DataFrames
- Use show_html(content) to render HTML content

CRITICAL RULES:
- DataFrames are PRE-LOADED in the environment. Use the variable names provided.
- Do NOT try to load files using pl.read_csv(). They are already in variables.
- Use show_table(df, "Title") to display DataFrames.
- Use alt.Chart(df)... and ASSIGN to 'chart' variable or just creating it is fine (we capture all Altair charts).
- Do not call .show() or .save().

POLARS SYNTAX:
- Use .group_by() instead of .groupby()"""

        prompt = f"""Generate Python code to answer this request: {prep_res["message"]}

Available DataFrames (pre-loaded): {var_list_str if var_list_str else 'None'}

Uploaded Files:
{files_desc}

{db_schema}

Generate only the Python code, wrapped in ```python``` blocks."""

        response = call_llm(prompt=prompt, system_prompt=system_prompt)
        code = parse_code_block(response)

        code = re.sub(r'\.show\s*\([^)]*\)', '', code)
        code = re.sub(r'\.groupby\s*\(', '.group_by(', code)

        return code

    def post(self, shared, prep_res, exec_res):
        shared["generated_code"] = exec_res
        return "default"


class ExecuteCodeNode(Node):
    """Executes generated code in the persistent kernel."""

    def prep(self, shared):
        code = shared.get("generated_code", "")
        kernel = shared.get("kernel") # SandboxKernel instance
        return {"code": code, "kernel": kernel}

    def exec(self, prep_res):
        kernel = prep_res["kernel"]
        code = prep_res["code"]
        
        if not kernel:
            return {"success": False, "error": "Kernel not initialized"}
            
        result = kernel.execute(code)
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
            artifacts = result.get("artifacts", {})
            
            prompt = f"""The user asked: {prep_res["user_message"]}

The following Python code was executed:
```python
{code}
```

And produced this output (which may contain artifact tags like ```table id``` or ```plot id```):
{output}

Respond to the user's request.
- Integrate the analysis findings into your response.
- IMPORTANT: The output contains tags like ```table ...``` and ```plot ...```. You MUST INCLUDE these tags in your final response where the table or plot should appear. Do not change them.
"""

            explanation = call_llm(prompt)

            return {
                "message": explanation,
                "code": code,
                "output": output if output else "Code executed successfully",
                "artifacts": artifacts,
                "error": None
            }
        else:
            error = result.get("error", "Unknown error")
            return {
                "message": f"I encountered an error while running the code: {error}",
                "code": code,
                "output": None,
                "artifacts": {},
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
        return prep_res

    def post(self, shared, prep_res, exec_res):
        if "chat_history" not in shared:
            shared["chat_history"] = []

        shared["chat_history"].append({
            "role": "user",
            "content": exec_res["user_message"]
        })

        response = exec_res["response"]
        assistant_content = response.get("message", "")
        
        shared["chat_history"].append({
            "role": "assistant",
            "content": assistant_content
        })

        return None
