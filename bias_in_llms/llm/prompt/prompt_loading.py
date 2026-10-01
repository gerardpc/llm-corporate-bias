"""This module loads prompt templates from a YAML file."""

import yaml


def load_prompt_templates(prompt_file_path: str) -> dict[str, str]:
    """
    Loads prompt templates from a YAML file.

    This function reads a YAML file specified by `prompt_file_path` and returns
    a dictionary of prompt templates. If the file is empty, not found, or contains
    invalid YAML, it returns an empty dictionary and prints a warning or error message.

    Args:
        prompt_file_path: The file path to the YAML file containing prompt templates.

    Returns:
        A dictionary where keys are template names and values are template strings.
        Returns an empty dictionary if the file is empty, not found,
        or contains invalid YAML.
    """
    try:
        with open(prompt_file_path) as f:
            prompt_templates = yaml.safe_load(f)
        if prompt_templates is None:  # Handles empty YAML
            print(
                f"Warning: Prompt file '{prompt_file_path}'is empty or not valid YAML.",
            )
            prompt_templates = {}  # Avoid NoneType errors later
        return prompt_templates
    except FileNotFoundError:
        print(f"Error: The prompt file '{prompt_file_path}' was not found.")
        return {}  # Return empty dict to prevent further errors
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{prompt_file_path}': {e}")
        return {}  # Return empty dict
