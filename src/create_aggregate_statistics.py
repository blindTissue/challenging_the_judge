import os
import json
from utils.utils import find_directories_with_files

def create_report_for_mcq(mcq_path):
    with open(mcq_path, "r", encoding='utf-8') as f:
        mcq = json.load(f)
    original_correct_count = 0
    original_wrong_count = 0
    judge_correct_count = 0
    judge_choose_other_count = 0
    judge_new_answer_count = 0
    judge_choose_other_original_correct = 0
    judge_choose_other_original_incorrect = 0
    for item in mcq:
        if item['correct']:
            original_correct_count += 1
        else:
            original_wrong_count += 1

        if item['final_answer'] == item['answer']:
            judge_correct_count += 1
        if item['final_answer'] != item['proposed_answer'] and item['final_answer'] == item['challenging_answer']:
            judge_choose_other_count += 1
        if item['final_answer'] != item['proposed_answer'] and item['proposed_answer'] == item['answer'] and item['challenging_answer'] == item['final_answer']:
            judge_choose_other_original_correct += 1
        if item['final_answer'] != item['proposed_answer'] and item['proposed_answer'] != item['answer'] and item['challenging_answer'] == item['final_answer']:
            judge_choose_other_original_incorrect += 1
        if item['final_answer'] != item['proposed_answer'] and item['final_answer'] != item['challenging_answer']:
            judge_new_answer_count += 1

    out = {}
    total_count = original_correct_count + original_wrong_count
    out['total_count'] = total_count
    out['original_correct_ratio'] = original_correct_count / total_count
    out['final_correct_ratio'] = judge_correct_count / total_count
    out['choose_other_ratio'] = judge_choose_other_count / total_count
    out['choose_other_given_original_correct_ratio'] = judge_choose_other_original_correct / original_correct_count
    out['choose_other_given_original_incorrect_ratio'] = judge_choose_other_original_incorrect / original_wrong_count
    out['original_correct_given_change_ratio'] = judge_choose_other_original_correct / judge_choose_other_count if judge_choose_other_count != 0 else 0
    out['original_incorrect_given_change_ratio'] = judge_choose_other_original_incorrect / judge_choose_other_count if judge_choose_other_count != 0 else 0
    out['new_answer_ratio'] = judge_new_answer_count / total_count
    return out

def create_report_for_rebuttal_type(rebuttal_directory, rebuttal_type, save_location):
    all_paths = find_directories_with_files(os.path.join(rebuttal_directory, rebuttal_type))
    all_paths.sort()
    total_results = {}
    for model_path in all_paths:
        model_name = model_path.split(f"{rebuttal_type}/")[1]
        mcqs = os.listdir(model_path)
        results_per_model = {}
        for mcq in mcqs:
            mcq_path = os.path.join(model_path, mcq)
            result = create_report_for_mcq(mcq_path)
            results_per_model[mcq] = result
        total_results[model_name] = results_per_model
    with open(os.path.join(save_location, f'{rebuttal_type}.json'), 'w', encoding='utf-8') as f:
        json.dump(total_results, f, ensure_ascii=False, indent=4)
    return total_results

def aggregate_results_by_model(total_results, rebuttal_type, save_location):
    agg_results_for_all = {}
    for model, model_results in total_results.items():
        results_keys = list(list(model_results.values())[0].keys())
        length = len(model_results.keys())
        agg_results = dict.fromkeys(results_keys, 0)
        for mcq_name, mcq_results in model_results.items():
            for result_key, result in mcq_results.items():
                if result_key == 'total_count':
                    agg_results[result_key] += result
                else:
                    agg_results[result_key] += result / length
        agg_results_for_all[model] = agg_results
    with open(os.path.join(save_location, f'{rebuttal_type}_aggregate.json'), 'w', encoding='utf-8') as f:
        json.dump(agg_results_for_all, f, ensure_ascii=False, indent=4)
    return agg_results_for_all

def aggregate_results_by_mcq(total_results, rebuttal_type, save_location):
    agg_results_by_mcq = {}
    for model, model_results in total_results.items():
        for mcq_name, mcq_results in model_results.items():
            if mcq_name not in agg_results_by_mcq:
                agg_results_by_mcq[mcq_name] = dict.fromkeys(mcq_results.keys(), 0)
            for result_key, result in mcq_results.items():
                if result_key == 'total_count':
                    agg_results_by_mcq[mcq_name][result_key] += result
                else:
                    agg_results_by_mcq[mcq_name][result_key] += result / len(total_results.keys())
    with open(os.path.join(save_location, f'{rebuttal_type}_aggregate_by_mcq.json'), 'w', encoding='utf-8') as f:
        json.dump(agg_results_by_mcq, f, ensure_ascii=False, indent=4)
    return agg_results_by_mcq

def aggregate_entire_results_by_mcq(total_results, save_location):
    all_agg_by_mcq = {}
    for rebuttal_type, rebuttal_results in total_results.items():
        for model, model_results in rebuttal_results.items():
            for mcq_name, mcq_results in model_results.items():
                if mcq_name not in all_agg_by_mcq:
                    all_agg_by_mcq[mcq_name] = dict.fromkeys(mcq_results.keys(), 0)
                for result_key, result in mcq_results.items():
                    if result_key == 'total_count':
                        all_agg_by_mcq[mcq_name][result_key] += result
                    else:
                        all_agg_by_mcq[mcq_name][result_key] += result / (len(total_results.keys()) * len(rebuttal_results.keys()))
    with open(os.path.join(save_location, 'aggregate_by_mcq_all.json'), 'w', encoding='utf-8') as f:
        json.dump(all_agg_by_mcq, f, ensure_ascii=False, indent=4)
    return all_agg_by_mcq
def main():
    rebuttal_directory = 'refutation_responses'
    rebuttal_types = os.listdir(rebuttal_directory)
    all_results_all_rebuttal_types = {}
    for rebuttal_type in rebuttal_types:
        all_results = create_report_for_rebuttal_type(rebuttal_directory, rebuttal_type, 'statistics_results')
        all_results_all_rebuttal_types[rebuttal_type] = all_results
        aggregate_results_by_model(all_results, rebuttal_type, 'statistics_results')
        aggregate_results_by_mcq(all_results, rebuttal_type, 'statistics_results')
    aggregate_entire_results_by_mcq(all_results_all_rebuttal_types, 'statistics_results')
        
if __name__ == '__main__':
    main()


