import datasets
import json

from datasets import load_dataset
dataset_path = "lighteval/MATH"
dataset_name = "math"

dataset = load_dataset(dataset_path, "all", split="test")
def extraxt_answer(solution):
    start = solution.find("\\boxed{")
    if start == -1:
        return None
    start += len("\\boxed{")
    # find the closing bracket
    level = 1
    end = start - 1
    try:
        while level > 0:
            end += 1
            if solution[end] == "{":
                level += 1
            elif solution[end] == "}":
                level -= 1
    except Exception as e:
        print(solution)
        print(start, end)
        print(solution[start:])
        print(solution[start:end])
        raise e
    return solution[start:end]

def shuffle_choices_and_create_example(row, index):
    new_example = {
        "id": index,
        "question": row['problem'],
        "problem": row['problem'],
        "solution": row['solution'],
        "level": row['level'],
        "type": row['type'],
        "answer": extraxt_answer(row['solution']),
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