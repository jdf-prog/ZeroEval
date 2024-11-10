import datasets
import json

dataset_path = "DongfuJiang/MATH-500"
dataset_name = "math_500"


dataset = datasets.load_dataset(dataset_path, split="test")

def shuffle_choices_and_create_example(row, index):
    new_example = {
        "id": row['unique_id'],
        "question": row['problem'],
        "problem": row['problem'],
        "solution": row['solution'],
        "answer": row['answer'],
        "source": dataset_path,
        "config": dataset_name,
        "task_type": "qa",
    }
    return new_example

dataset = dataset.map(shuffle_choices_and_create_example, with_indices=True, remove_columns=dataset.column_names)
dataset.push_to_hub(
    repo_id="DongfuJiang/zeroeval",
    config_name=dataset_name,
    split='test',
    commit_message=f"Add {dataset_name} dataset",
)