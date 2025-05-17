import os

def format_problem(problem: dict) -> str:
    """
    Formats a problem dictionary into a string representation.
    """
    
    question = problem['question']
    choices = problem['choices']
    answer_start = "A"
    choices_str = "\n".join(
        f"{chr(ord(answer_start) + i)}. {choice}" for i, choice in enumerate(choices)
    )
    prompt = f"Question: {question}\n\n{choices_str}"
    return prompt



def find_directories_with_files(start_dir):
    """
    Recursively walks through the directory and its subdirectories,
    returning a list of directories that contain at least one file.

    Args:
        start_dir (str): The path to the starting directory.

    Returns:
        list: A list of absolute paths to directories containing at least one file.
    """
    directories_with_files = []
    for root, _, files in os.walk(start_dir):
        meaningful_files = [f for f in files if not f.startswith(".")]
        if meaningful_files:

            directories_with_files.append(os.path.abspath(root))
    return directories_with_files