import os
import pathlib
from datetime import datetime
import re
import json
import argparse
import multiprocessing
from unittest import case

from together import Together
from functools import partial
from groq import Groq
from openai import OpenAI
from utils.utils import format_problem
from tqdm import tqdm


MCQ_PATH = "mcq_data"

def ask_llm(problem_text, prompt_template, client, model):
    ask_prompt = prompt_template.format(problem_text=problem_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages = [
                {"role": "user", "content": ask_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")

def create_api_client(api_provider: str):
    match api_provider:
        case "openai":
            return OpenAI()
        case "together":
            return Together()
        case "groq":
            return Groq()
        case _:
            raise Exception(f"Unknown api provider: {api_provider}")


def extract_answer_and_reasoning(response_text: str):
    pattern = r'(.*?)(?:ANSWER|Answer):\s*(?:\(|\[)?([A-J])(?:\)|\])?(?:\s|\.|\n|$)'
    answer_match = re.search(pattern, response_text, re.DOTALL)
    final_answer = None
    reasoning = response_text

    if answer_match:
        final_answer = answer_match.group(2)
        reasoning = answer_match.group(1)
    return final_answer, reasoning



def process_problem(problem: dict, model: str, api_provider: str, prompt_template: str) -> dict:
    """
    Creates Worker function processing a single problem.
    Args:
        prompt_template: template for asking question to llm
        problem: Problem given as dict.
        model: LLM Model
        api_provider: api provider. Currently GROQ, OpenAI or Together

    Returns: Dictionary containing question, answer, choices, response, and proposed answer.
    """
    problem_text = format_problem(problem)
    client = create_api_client(api_provider)
    problem_answer = problem["answer"]
    answer_alphabet = chr(ord('A') + problem_answer)

    try:
        response = ask_llm(problem_text, prompt_template, client, model)
        final_answer, reasoning = extract_answer_and_reasoning(response)

        return {
            "question": problem['question'],
            "answer": answer_alphabet,
            "choices": problem["choices"],
            "response": response,
            "proposed_answer": final_answer
        }

    except Exception as e:
        print(f"Error processing question: {problem['question']}, {str(e)}")
        return {
            "question": problem['question'],
            "answer": answer_alphabet,
            "choices": problem["choices"],
            "response": f"Error: {str(e)}",
            "proposed_answer": None
        }

def answer_questions_on_a_json(json_filename: str, model: str, api_provider: str, prompt_template: str, save_directory:str):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    json_path = os.path.join(MCQ_PATH, json_filename)

    with open(json_path, "r") as f:
        print(f"Starting to process {json_path}")
        problems = json.load(f)

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)

    worker_func = partial(process_problem, model=model, api_provider=api_provider, prompt_template=prompt_template)

    results = list(tqdm(pool.imap(worker_func, problems), total=len(problems)))
    pool.close()
    pool.join()

    save_path = os.path.join(save_directory, json_filename)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':

    question_prompt_template_path = os.path.join("prompt_templates", "mcq_question_prompt_template.txt")
    question_prompt_template = open(question_prompt_template_path).read()

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--mcq_to_run", type=str, default="all")
    parser.add_argument("--save-path", type=str, default="llm_responses")

    args = parser.parse_args()


    save_path = os.path.join(args.save_path, args.model)

    if args.mcq_to_run == "all":
        all_mcqs = os.listdir(MCQ_PATH)
        all_mcqs.sort()

        for mcq in all_mcqs:
            print(f"Processing mcq: {mcq}")
            answer_questions_on_a_json(mcq, args.model, args.api_provider, question_prompt_template, save_path)

    else:
        if not args.mcq_to_run.endswith(".json"):
            args.mcq_to_run = args.mcq_to_run + ".json"
        answer_questions_on_a_json(args.mcq_to_run, args.model, args.api_provider, question_prompt_template, save_path)



