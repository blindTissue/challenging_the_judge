import re
import json
import argparse
import multiprocessing
import os

from together import Together
from functools import partial
from groq import Groq
from openai import OpenAI
from utils.utils import format_problem, find_directories_with_files
from tqdm import tqdm
from utils.refutation_formatter import *
from utils.mcq_prompter import format_problem_into_mcq_for_llm
import time
import argparse



def send_refuting_statement(problem, client, model, refutation_type="are_you_sure_with_answer",
                            max_retries=5, initial_delay=1.0, max_delay = 60.0):
    mcq_prompt = format_problem_into_mcq_for_llm(problem)
    original_response = problem['response']

    if refutation_type == "are_you_sure_with_answer":
        refutation = send_are_you_sure_with_answer(problem)
    elif refutation_type == "all_reasoning_refutation":
        refutation = send_all_reasoning_refutation(problem)
    elif refutation_type == "truncated_refutation":
        refutation = send_truncated_reasoning_refutation(problem)
    elif refutation_type == "send_sure_rebuttal":
        refutation = send_sure_refutation(problem)
    elif refutation_type == "divergence_rebuttal":
        refutation, divergence = get_divergence_feedback(problem, model, client)
    elif refutation_type == 'only_answer_refutation':
        refutation = send_only_answer_refutation(problem)
    else:
        print(f"Unknown refutation type: {refutation_type}")  # Added f-string for clarity
        # It might be better to return None or raise an error here
        # instead of proceeding, to avoid issues later.
        return None, None  # Example: return None if type is unknown

    messages = [
        {"role": "user", "content": mcq_prompt},
        {"role": "assistant", "content": original_response},
        {"role": "user", "content": refutation}
    ]

    current_delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0
            )
            if refutation_type == "divergence_rebuttal":
                return response.choices[0].message.content, (refutation, divergence)
            return response.choices[0].message.content, refutation

        except Exception as e:
            # Catch any other unexpected errors.
            # Check if it's a 503 based on the string if no specific exception was caught
            if ("503" in str(e) or "server is overloaded" in str(e).lower()) and attempt < max_retries - 1 :
                print(f"API call failed with generic error (likely retriable): {e}. Retrying in {current_delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(current_delay + random.uniform(0, current_delay * 0.1)) # Add jitter
                current_delay = min(current_delay * 2, max_delay) # Exponential backoff, capped
            else:
                # If it's not a recognized retriable error or last attempt, re-raise
                raise Exception(f"API call failed with unexpected error: {str(e)}")

    # Should not be reached if max_retries > 0, but as a fallback:
    raise Exception(f"API call failed after {max_retries} attempts (exhausted retries).")



def extract_answer(response_text):
    if response_text[-2:] == "**":
        return response_text[-3], response_text
    # Extract the final answer using regex
    pattern = r'(.*?)(?:ANSWER|Answer):\s*(?:\(|\[)?([A-J])(?:\)|\])?(?:\s|\.|\n|$)'
    answer_match = re.search(pattern, response_text, re.DOTALL)

    final_answer = None
    reasoning = response_text

    if answer_match:
        final_answer = answer_match.group(2)
        reasoning = response_text[:answer_match.start()].strip()

    return final_answer, reasoning

def process_problem(problem, model, refutation_type):
    if model[:3] == 'gpt':
        client = OpenAI()
    elif 'Scout' in model:
        client = Groq()
    else:
        client = Together()

    try:
        response, refutation = send_refuting_statement(problem, client, model, refutation_type=refutation_type)
        final_answer, reasoning = extract_answer(response)
        problem_copy = problem.copy()
        problem_copy['response_to_refute'] = response
        problem_copy['refutation'] = refutation
        problem_copy['final_answer'] = final_answer

        return problem_copy

    except Exception as e:
        print(f"API call failed: {str(e)}")
        problem_copy = problem.copy()
        problem_copy['response_to_refute'] = None
        problem_copy['refutation'] = None
        problem_copy['final_answer'] = None
        return problem_copy

def run_refutation_on_file(refutation_path, save_path, refutation_type):
    print(refutation_path)
    with open(refutation_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    model_name = problems[0]['llm_name']
    file_name = os.path.basename(refutation_path)

    save_directory = os.path.join(save_path, refutation_type, model_name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    output_path = os.path.join(save_directory, file_name)
    if os.path.isfile(output_path):
        print(f"Output file already exists: {output_path}")
        return

    num_processes = multiprocessing.cpu_count()
    if "DeepSeek-V3" in model_name:
        num_processes = 5 # repetitvely getting errors.
    num_processes = 15
    pool = multiprocessing.Pool(processes = num_processes)
    worker_func = partial(process_problem, model=model_name, refutation_type=refutation_type)
    results = list(tqdm(pool.imap(worker_func, problems), total = len(problems)))
    pool.close()
    pool.join()

    print("Saving results to", output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def main(args):
    base_directory = "llm_refutations"
    save_path = "refutation_responses"
    refutation_type = args.refutation_type
    all_response_directories = find_directories_with_files(base_directory)
    print(all_response_directories)
    for all_response_directory in all_response_directories:
        print("Processing", all_response_directory)
        mcqs = os.listdir(all_response_directory)
        # remove .DS_Store
        mcqs.sort()
        mcqs = [mcq for mcq in mcqs if mcq != ".DS_Store"]

        for mcq in mcqs:
            problem_path = os.path.join(all_response_directory, mcq)
            run_refutation_on_file(problem_path, save_path, refutation_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refutation-type", default="divergence_rebuttal")
    args = parser.parse_args()
    main(args)




