import re
import json
import argparse
import multiprocessing
import os

from google.auth.environment_vars import GOOGLE_CLOUD_QUOTA_PROJECT
from together import Together
from functools import partial
from groq import Groq
from openai import OpenAI
from utils.utils import format_problem, find_directories_with_files
from tqdm import tqdm

judge_prompt_template_path = os.path.join('prompt_templates', 'judge_template.txt')
JUDGE_TEMPLATE = open(judge_prompt_template_path).read()


def get_response_llm_as_judge(problem, client, model):
    question_text = format_problem(problem)
    response_1 = problem['response']
    response_2 = problem['challenging_response']
    judge_prompt = JUDGE_TEMPLATE.format(question_text=question_text, response_1=response_1, response_2=response_2)

    try:
        response = client.chat.completions.create(
            model=model,
            messages = [
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")


def extract_answer(response_text):
    if response_text[-2:] == "**":
        return response_text[-3], response_text
    # Extract the final answer using regex
    pattern = r'(.*?)(?:ANSWER|Answer):\s*(?:\(|\[)?([A-J])(?:\)|\])?(?:\s|\.|\n|$)'
    answer_match = re.search(pattern, response_text, re.DOTALL)

    # Default values if no match is found
    final_answer = None
    reasoning = response_text

    if answer_match:
        # Group 2 contains the letter (A, B, C, or D)
        final_answer = answer_match.group(2)
        # Get the reasoning (everything before "ANSWER:")
        reasoning = response_text[:answer_match.start()].strip()

    return final_answer, reasoning

def process_problem(problem, model):
    if model[:3] == 'gpt':
        client = OpenAI()
    if 'Scout' in model:
        client = Groq()
    else:
        client = Together()

    try:
        response = get_response_llm_as_judge(problem, client, model)
        final_answer, reasoning = extract_answer(response)
        problem_copy = problem.copy()
        problem_copy['judge_response'] = response
        problem_copy['judge_final_answer'] = final_answer
        return problem_copy

    except Exception as e:
        print(f"API call failed: {str(e)}")
        problem_copy = problem.copy()
        problem_copy['judge_response'] = None
        problem_copy['judge_final_answer'] = None
        return problem_copy

def run_judge_on_file(refutation_path, save_path):
    print(refutation_path)
    with open(refutation_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    model_name = problems[0]['llm_name']
    file_name = os.path.basename(refutation_path)

    save_directory = os.path.join(save_path, model_name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    output_path = os.path.join(save_directory, file_name)
    if os.path.isfile(output_path):
        print(f"Output file already exists: {output_path}")
        return

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = num_processes)
    worker_func = partial(process_problem, model=model_name)
    results = list(tqdm(pool.imap(worker_func, problems), total = len(problems)))
    pool.close()
    pool.join()

    print("Saving results to", output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def main():
    base_directory = "llm_refutations"
    save_path = "llm_refutation_judge"
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
            run_judge_on_file(problem_path, save_path)

if __name__ == "__main__":
    main()




