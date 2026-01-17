"""Flow orchestration for the analytical chatbot."""

from pocketflow import Flow
from utils.nodes import (
    GetInputNode,
    ClassifyIntentNode,
    ConversationResponseNode,
    GenerateCodeNode,
    ExecuteCodeNode,
    FormatResultsNode,
    OutputResponseNode
)


def create_chatbot_flow() -> Flow:
    """
    Create and return the chatbot flow.
    """
    get_input = GetInputNode()
    classify_intent = ClassifyIntentNode(max_retries=3, wait=1)
    conversation_response = ConversationResponseNode(max_retries=2, wait=1)
    generate_code = GenerateCodeNode(max_retries=3, wait=1)
    execute_code = ExecuteCodeNode()
    format_results = FormatResultsNode(max_retries=2, wait=1)
    output_response = OutputResponseNode()

    get_input >> classify_intent

    classify_intent - "conversation" >> conversation_response
    conversation_response - "output" >> output_response

    classify_intent - "code_execution" >> generate_code
    generate_code >> execute_code
    execute_code >> format_results
    format_results >> output_response
    # Retry code generation when execution fails (see ExecuteCodeNode.post).
    execute_code - "retry" >> generate_code

    get_input - "output" >> output_response

    return Flow(start=get_input)


def run_chatbot(user_message: str, kernel = None, uploaded_files: dict = None, chat_history: list = None) -> dict:
    """
    Run the chatbot flow with user input.

    Args:
        user_message: The user's message
        kernel: The SandboxKernel instance (multiprocessing)
        uploaded_files: Dict of filename -> metadata (actual data is in kernel)
        chat_history: List of previous messages

    Returns:
        Dict containing the response with message, code, output, plots, error
    """
    shared = {
        "user_message": user_message,
        "kernel": kernel,
        "uploaded_files": uploaded_files or {},
        "chat_history": chat_history or [],
        "response": {
            "message": "",
            "code": None,
            "output": None,
            "artifacts": {},
            "error": None
        }
    }

    flow = create_chatbot_flow()
    flow.run(shared)

    return {
        "response": shared.get("response", {}),
        "chat_history": shared.get("chat_history", [])
    }
