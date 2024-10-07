import datasets
import json

dataset_path = "lighteval/MATH"
dataset_name = "math"

filter_key = None
# filter_key = "Level 5"

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

dataset = datasets.load_dataset(dataset_path, name="all", split="test", trust_remote_code=True)
if filter_key:
    dataset = dataset.filter(lambda x: filter_key in x['level'])

def shuffle_choices_and_create_example(row, index):
    answer = extraxt_answer(row['solution'])
    if not answer:
        print(f"Cannot extract answer from {row['solution']}")
    new_example = {
        "id": f"{dataset_name}_{index}",
        "question": row['problem'],
        "answer": answer,
        "source": dataset_path,
        "config": "all_test",
        "task_type": "qa",
        "level": row['level'],
    }
    
    return new_example

dataset = dataset.map(shuffle_choices_and_create_example, with_indices=True, remove_columns=dataset.column_names)
dataset.push_to_hub(
    repo_id="DongfuJiang/zeroeval",
    config_name=dataset_name,
    split='test',
    commit_message=f"Add {dataset_name} dataset",
)