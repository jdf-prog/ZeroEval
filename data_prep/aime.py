
import datasets
import json

dataset_path = "qq8933/AIME_1983_2024"
dataset_name = "aime_2024_II"
filter_key = "2024-II"


dataset = datasets.load_dataset(dataset_path, split="train")
dataset = dataset.filter(lambda x: filter_key in x['ID'])

def shuffle_choices_and_create_example(row, index):
    new_example = {
        "id": row['ID'],
        "question": row['Question'],
        "answer": row['Answer'],
        "source": dataset_path,
        "config": None,
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