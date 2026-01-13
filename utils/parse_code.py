"""Utility to parse code blocks from LLM responses."""

import re


def parse_code_block(text: str) -> str:
    """
    Extract Python code from an LLM response.

    Handles:
    - ```python ... ``` blocks
    - ``` ... ``` blocks (generic)
    - Raw code if no fence found

    Args:
        text: The LLM response text

    Returns:
        Extracted Python code
    """
    # Try to find ```python ... ``` block
    python_pattern = r"```python\s*(.*?)\s*```"
    match = re.search(python_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find generic ``` ... ``` block
    generic_pattern = r"```\s*(.*?)\s*```"
    match = re.search(generic_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No code block found, return the text as-is (might be raw code)
    return text.strip()


if __name__ == "__main__":
    # Test cases
    test_cases = [
        """Here's the code:
```python
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
print(df.mean())
```
This will calculate the mean.""",
        """```
x = 5
print(x * 2)
```""",
        "print('hello world')",
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"Input: {test[:50]}...")
        print(f"Output: {parse_code_block(test)}")
        print("-" * 40)
