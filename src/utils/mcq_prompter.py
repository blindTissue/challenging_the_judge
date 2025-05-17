import os
from utils.utils import format_problem
MCQ_QUESTION_TEMPLATE_PATH = os.path.join('prompt_templates', 'mcq_question_prompt_template.txt')
MCQ_QUESTION_TEMPLATE = open(MCQ_QUESTION_TEMPLATE_PATH).read()

def format_problem_into_mcq_for_llm(problem):
    problem_text = format_problem(problem)
    out = MCQ_QUESTION_TEMPLATE.format(problem_text=problem_text)
    return out