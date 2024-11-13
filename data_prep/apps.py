# https://huggingface.co/datasets/codeparrot/apps

import datasets
import json

dataset_path = "codeparrot/apps"
dataset_name = "apps"


dataset = datasets.load_dataset(dataset_path, "all", split="test")

def shuffle_choices_and_create_example(row, index):
    new_example = {
        "id": row['problem_id'],
        "question": row['question'],
        "problem": row['question'],
        "answer": row['solutions'],
        "difficulty": row['difficulty'],
        "testcases": row['input_output'],
        "source": dataset_path,
        "config": dataset_name,
        "task_type": "code_completion",
    }
    return new_example

dataset = dataset.map(shuffle_choices_and_create_example, with_indices=True, remove_columns=dataset.column_names)
dataset.push_to_hub(
    repo_id="DongfuJiang/zeroeval",
    config_name=dataset_name,
    split='test',
    commit_message=f"Add {dataset_name} dataset",
)