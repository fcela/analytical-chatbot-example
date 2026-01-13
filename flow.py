"""Flow orchestration for the analytical chatbot."""

from pocketflow import Flow
from nodes import (
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

    Flow structure:
    ```
    GetInput -> ClassifyIntent --conversation--> ConversationResponse -> Output
                     |
                     +--code_execution--> GenerateCode -> ExecuteCode -> FormatResults -> Output
    ```
    """
    # Create node instances
    get_input = GetInputNode()
    classify_intent = ClassifyIntentNode(max_retries=3, wait=1)
    conversation_response = ConversationResponseNode(max_retries=2, wait=1)
    generate_code = GenerateCodeNode(max_retries=3, wait=1)
    execute_code = ExecuteCodeNode()
    format_results = FormatResultsNode(max_retries=2, wait=1)
    output_response = OutputResponseNode()

    # Wire the flow
    # Main path: input -> classify
    get_input >> classify_intent

    # Branch 1: conversation path
    classify_intent - "conversation" >> conversation_response
    conversation_response - "output" >> output_response

    # Branch 2: code execution path
    classify_intent - "code_execution" >> generate_code
    generate_code >> execute_code
    execute_code >> format_results
    format_results >> output_response

    # Handle edge case: empty input goes directly to output
    get_input - "output" >> output_response

    return Flow(start=get_input)


def run_chatbot(user_message: str, uploaded_files: dict = None, chat_history: list = None) -> dict:
    """
    Run the chatbot flow with user input.

    Args:
        user_message: The user's message
        uploaded_files: Dict of filename -> DataFrame for uploaded files
        chat_history: List of previous messages

    Returns:
        Dict containing the response with message, code, output, plots, error
    """
    shared = {
        "user_message": user_message,
        "uploaded_files": uploaded_files or {},
        "chat_history": chat_history or [],
        "response": {
            "message": "",
            "code": None,
            "output": None,
            "plots": None,
            "error": None
        }
    }

    flow = create_chatbot_flow()
    flow.run(shared)

    return {
        "response": shared.get("response", {}),
        "chat_history": shared.get("chat_history", [])
    }


if __name__ == "__main__":
    # Test the flow
    import os

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Set it to test the flow.")
        print("Example: export OPENAI_API_KEY='your-key-here'")
    else:
        # Test conversation
        print("Testing conversation flow...")
        result = run_chatbot("Hello! What can you do?")
        print(f"Response: {result['response']['message']}")
        print()

        # Test code execution (if pandas is available)
        try:
            import pandas as pd
            print("Testing code execution flow...")
            test_df = pd.DataFrame({
                'name': ['Alice', 'Bob', 'Charlie'],
                'age': [25, 30, 35],
                'salary': [50000, 60000, 70000]
            })
            result = run_chatbot(
                "What is the average age and salary?",
                uploaded_files={"employees.csv": test_df}
            )
            print(f"Response: {result['response']['message']}")
            print(f"Code: {result['response']['code']}")
            print(f"Output: {result['response']['output']}")
        except ImportError:
            print("Pandas not available, skipping code execution test")
