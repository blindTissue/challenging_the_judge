# file to run items which had API error call problem.
# Also checks for **Answer B**
import re
from utils.utils import find_directories_with_files
import os
import json


def extract_answer_boldface(response_blob):
    """
    Some of the more recent llms have boldface answers. Simple program to remove it.
    Args:
        response_blob:

    Returns:

    """
    if response_blob['proposed_answer'] == None:
        llm_response = response_blob["response"]
        if llm_response[-2:] == "**":
            return llm_response[-3]
    return None


if __name__ == "__main__":
    root_dir = "llm_responses"
    all_paths = find_directories_with_files(root_dir)
    for model_path in all_paths:
        categories = os.listdir(model_path)
        for category in categories:
            with open(os.path.join(model_path, category), 'r', encoding='utf-8') as f:
                responses = json.load(f)
            for response in responses:
                change = extract_answer_boldface(response)
                if change is not None:
                    response['proposed_answer'] = change

            with open(os.path.join(model_path, category), 'w', encoding='utf-8') as f:
                json.dump(responses, f, ensure_ascii=False, indent=4)






