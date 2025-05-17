import datasets
import random
import json
import os


def get_mmlu(save_path: str, sample_count=300):
    """
    gets MMLU from the hub and saves a sample of it to the specified path.
    
    Args:
        save_path (str): The path where the dataset will be saved.
        sample_count (int): The number of samples to load from the dataset.
    """
    
    # uf save path does not exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_path_file= os.path.join(save_path, "mmlu_sample.json")
    
    # if file already exists, exit
    if os.path.exists(save_path_file):
        print(f"File {save_path_file} already exists. Skipping download.")
        return
    
    mmlu = datasets.load_dataset("cais/mmlu", "all")
    mmlu = mmlu['test']
    mmlu_features = mmlu.features.keys()
    random_indices = random.sample(range(len(mmlu)), sample_count)
    mmlu_sample = mmlu.select(random_indices)
    out = []
    for i in range(len(mmlu_sample)):
        item = {}
        for feature in mmlu_features:
            item[feature] = mmlu_sample[i][feature]
        out.append(item)
    with open(save_path_file, 'w', encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
    print(f"Saved {sample_count} MMLU samples to {save_path_file}")
    
    
def get_mmlu_pro(save_path: str, sample_count=300):
    """
    Gets MMLU-Pro from the hub and saves a sample of it to the specified path.
    
    Args:
        save_path (str): The path where the dataset will be saved.
        sample_count (int): The number of samples to load from the dataset.
    """
    
    # If save path does not exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    
    save_path_file = os.path.join(save_path, "mmlu_pro_sample.json")
    
    if os.path.exists(save_path_file):
        print(f"File {save_path_file} already exists. Skipping download.")
        return
    
    mmlu_pro = datasets.load_dataset('TIGER-Lab/MMLU-Pro')
    mmlu_pro = mmlu_pro['test']
    mmlu_pro_features = mmlu_pro.features.keys()
    random_indices = random.sample(range(len(mmlu_pro)), sample_count)
    mmlu_pro_sample = mmlu_pro.select(random_indices)
    out = []
    for i in range(len(mmlu_pro_sample)):
        item = {}
        for feature in mmlu_pro_features:
            if feature == "answer":
                item["letter_answer"] = mmlu_pro_sample[i][feature]
            elif feature == "answer_index":
                item["answer"] = mmlu_pro_sample[i][feature]
            elif feature == "options":
                item['choices'] = mmlu_pro_sample[i][feature]
            else:
                item[feature] = mmlu_pro_sample[i][feature]
        out.append(item)
    with open(save_path_file, 'w', encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
    print(f"Saved {sample_count} MMLU-Pro samples to {save_path_file}")
    
def get_sciq(save_path: str, sample_count = 300):
    """
    gets sciq from the hub and saves a sample of it to the specified path.
    Args:
        save_path (str): The path where the dataset will be saved.
        sample_count (int): The number of samples to load from the dataset.
    """
    
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
        
    save_path_file = os.path.join(save_path, "sciq_sample.json")
    
    if os.path.exists(save_path_file):
        print(f"File {save_path_file} already exists. Skipping download.")
        return
    
    sciq = datasets.load_dataset("allenai/sciq")
    sciq = sciq['test']
    random_indices = random.sample(range(len(sciq)), sample_count)
    sciq_sample = sciq.select(random_indices)
    out = []
    for i, v in enumerate(sciq_sample):
        item = {}
        choices, index = format_choices_and_answer_for_sciq(v)
        item['question'] = v['question']
        item['choices'] = choices
        item['answer'] = index
        item['support'] = v['support']
        out.append(item)
    with open(save_path_file, 'w', encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
        

def format_choices_and_answer_for_sciq(item):
    choices = [item['distractor1'], item['distractor2'], item['distractor3'], item['correct_answer']]
    answer = item['correct_answer']
    random.shuffle(choices)
    correct_index = choices.index(answer)
    return choices, correct_index

def get_arc_challenge(save_path: str, sample_count=300):
    """
    Gets ARC-Challenge from the hub and saves a sample of it to the specified path.
    
    Args:
        save_path (str): The path where the dataset will be saved.
        sample_count (int): The number of samples to load from the dataset.
    """
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_path_file = os.path.join(save_path, "arc_challenge_sample.json")
    
    if os.path.exists(save_path_file):
        print(f"File {save_path_file} already exists. Skipping download.")
        return
    arc_challenge = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge")
    arc_challenge = arc_challenge['test']
    random_indices = random.sample(range(len(arc_challenge)), sample_count)
    arc_challenge_sample = arc_challenge.select(random_indices)
    out = []
    for i, v in enumerate(arc_challenge_sample):
        item = {}
        choices, index = format_choices_and_answer_for_arc_challenge(v)
        item['question'] = v['question']
        item['choices'] = choices
        item['answer'] = index
        out.append(item)
    with open(save_path_file, 'w', encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
    

def get_arc_easy(save_path: str, sample_count=300):
    """
    Gets ARC-Easy from the hub and saves a sample of it to the specified path.
    
    Args:
        save_path (str): The path where the dataset will be saved.
        sample_count (int): The number of samples to load from the dataset.
    """
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_path_file = os.path.join(save_path, "arc_easy_sample.json")
    if os.path.exists(save_path_file):
        print(f"File {save_path_file} already exists. Skipping download.")
        return
    arc_easy = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy")
    arc_easy = arc_easy['test']
    random_indices = random.sample(range(len(arc_easy)), sample_count)
    arc_easy_sample = arc_easy.select(random_indices)
    out = []
    for i, v in enumerate(arc_easy_sample):
        item = {}
        choices, index = format_choices_and_answer_for_arc_challenge(v)
        item['question'] = v['question']
        item['choices'] = choices
        item['answer'] = index
        out.append(item)
    with open(save_path_file, 'w', encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
        
def format_choices_and_answer_for_arc_challenge(problem):
    possible_choices = problem['choices']['text']
    labels = problem['choices']['label']
    answer = problem['answerKey']
    label_start = labels[0]
    answer_index = ord(answer) - ord(label_start)
    return possible_choices, answer_index


def get_medmcqa(save_path: str, sample_count=300):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_path_file = os.path.join(save_path, "medmcqa_sample.json")
    
    medmcqa = datasets.load_dataset("openlifescienceai/medmcqa")
    
    if os.path.exists(save_path_file):
        print(f"File {save_path_file} already exists. Skipping download.")
        return
    
    medmcqa = medmcqa['train'] # test doesn't have ground truth
    random_indices = random.sample(range(len(medmcqa)), sample_count)
    medmcqa_sample = medmcqa.select(random_indices)
    out = []
    for i,v in enumerate(medmcqa_sample):
        item = {}
        item['question'] = v['question']
        choices, answer = format_choices_and_answer_for_medmcqa(v)
        item['choices'] = choices
        item['answer'] = answer
        out.append(item)
    with open(save_path_file, 'w', encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
        
        
def format_choices_and_answer_for_medmcqa(problem):
    choices = [problem['opa'], problem['opb'], problem['opc'], problem['opd']]
    answer = problem['cop']
    return choices, answer



def get_logiqa(save_path: str, sample_count=300):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_file = os.path.join(save_path, "logiqa_sample.json")
    logiqa = datasets.load_dataset("lucasmccabe/logiqa")
    
    if os.path.exists(save_path_file):
        print(f"File {save_path_file} already exists. Skipping download.")
        return
    
    logiqa = logiqa['train']
    random_indices = random.sample(range(len(logiqa)), sample_count)
    logiqa_sample = logiqa.select(random_indices)
    out = []
    for i, v in enumerate(logiqa_sample):
        item = {}
        question = v['context'] + " " + v['query']
        item['question'] = question
        item['choices'] = v['options']
        item['answer'] = v['correct_option']
        out.append(item)
    with open(save_path_file, 'w', encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)

def get_commonsense_qa(save_path: str, sample_count=300):
    """
    Gets CommonSenseQA from the hub and saves a sample of it to the specified path.
    
    Args:
        save_path (str): The path where the dataset will be saved.
        sample_count (int): The number of samples to load from the dataset.
    """
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_path_file = os.path.join(save_path, "commonsense_qa_sample.json")
    
    if os.path.exists(save_path_file):
        print(f"File {save_path_file} already exists. Skipping download.")
        return
    
    commonsense_qa = datasets.load_dataset("tau/commonsense_qa")
    commonsense_qa = commonsense_qa['train']
    random_indices = random.sample(range(len(commonsense_qa)), sample_count)
    commonsense_qa_sample = commonsense_qa.select(random_indices)
    out = []
    for i, v in enumerate(commonsense_qa_sample):
        item = {}
        item['question'] = v['question']
        item['choices'] = v['choices']['text']
        item['answer'] = ord(v['answerKey']) - ord('A')
        out.append(item)
        
    with open(save_path_file, 'w', encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
    

# Currently passing since hellaswag's prompt would be different from other mcqs.
def get_hellaswag(save_path: str, sample_count=300):
    pass
    

def get_data(save_path: str, sample_count=300) -> None:
    """
    Gets all datasets and saves samples to the specified path.
    
    Args:
        save_path (str): The path where the datasets will be saved.
        sample_count (int): The number of samples to load from each dataset.
    """
    get_mmlu(save_path, sample_count)
    get_mmlu_pro(save_path, sample_count)
    get_sciq(save_path, sample_count)
    get_arc_challenge(save_path, sample_count)
    get_arc_easy(save_path, sample_count)
    get_medmcqa(save_path, sample_count)
    get_commonsense_qa(save_path, sample_count)
    get_logiqa(save_path, sample_count)
    



if __name__ == "__main__":

    save_path = "mcq_data"
    sample_count = 300
    get_data(save_path, sample_count)