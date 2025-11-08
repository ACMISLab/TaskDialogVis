import json
import re
from tqdm import tqdm
from openai import OpenAI
import utils

# Initialize OpenAI client (custom local server setup)
client = OpenAI(api_key="adsa", base_url="http://10.10.10.153:20048/v1")


def filter_step_answer(answer: str):
    """
    Extracts all <answer>...</answer> contents for 7 reasoning steps from a generated string.

    Args:
        answer (str): The raw model output containing <step i> and <answer> tags.

    Returns:
        list[str]: A list of 7 extracted answers corresponding to each reasoning step.
    """
    ans_list = []
    temp = answer

    for i in range(7):
        # Extract the substring before </step i>
        before = temp.split(f" </step {i + 1}>")[0]
        if i != 6:
            # Move the cursor forward for the next step
            temp = temp.split(f" </step {i + 1}")[1]

        # Cleanly extract the answer text between <answer> and </answer>
        before = before.split(" </answer>")[0]
        ans = before.split("<answer> ")[1]
        ans_list.append(ans)

    return ans_list


def filter_step(step_num: int, output_str: str):
    """
    Extracts the content of a specific <step n>...</step n> block.

    Args:
        step_num (int): Step number to extract.
        output_str (str): Full reasoning string.

    Returns:
        str: Content of the given step.
    """
    before = output_str.split(f"<step {step_num}> ")[1]
    step = before.split(f" </step {step_num}>")[0]
    return step


def get_dpo_data(step_num: int, chosen_str: str, rejected_str: str):
    """
    Builds a DPO-style training sample from a reasoning mismatch.

    Args:
        step_num (int): The step index where reasoning diverged.
        chosen_str (str): The correct (reference) reasoning output.
        rejected_str (str): The incorrect (model) reasoning output.

    Returns:
        dict: Dictionary with prompt context and step-level comparison data.
    """
    chosen_step = filter_step(step_num, chosen_str)
    rejected_step = filter_step(step_num, rejected_str)

    # Prefix includes reasoning steps before the divergence
    prefix = rejected_str.split(f"<step {step_num}> ")[0]
    prefix += f"<step {step_num}>"

    return {
        "initial_reason_steps": prefix,
        "chosen": chosen_step,
        "rejected": rejected_step,
    }


# Load ground-truth reasoning path data
with open(r"truth_reasoning_path.json", "r", encoding="utf-8") as file:
    train_data = json.load(file)

# Accumulator for DPO preference samples
preference_data = []

# Process each data entry with progress bar
for index, data in enumerate(tqdm(train_data)):
    retry_count = 0

    while retry_count < 3:
        try:
            # Step 1: Get the correct answer decomposition
            true_answer = filter_step_answer(data["output"])

            # Step 2: Generate model response
            messages = [{"role": "user", "content": data["instruction"]}]
            response = client.chat.completions.create(
                model="deepseek",
                messages=messages,
                stream=False,
                temperature=0.7,
                top_p=0.95,
            ).choices[0].message.content

            print(response)
            response_answer = filter_step_answer(response)

            # Skip incomplete outputs
            if len(response_answer) != 7:
                break

            print(response_answer)
            print(true_answer)

            # Step 3: Compare step-by-step reasoning
            for i in range(7):
                # === Type 1: Direct string comparison ===
                if i in [0, 2, 3] and response_answer[i].replace(" ", "") != true_answer[i].replace(" ", ""):
                    print(f"Mismatch at step {i}")
                    dpo_item = get_dpo_data(i + 1, data["output"], response)
                    dpo_item["prompt"] = data["instruction"]
                    preference_data.append(dpo_item)
                    with open(r"preference_data/preference_data.json", 'w', encoding='utf-8') as f:
                        json.dump(preference_data, f, ensure_ascii=False, indent=4)
                    break

                # === Type 2: Dictionary equality (with sorted lists) ===
                elif i == 1:
                    dict1 = json.loads(response_answer[i].replace("'", "\"").replace("True", "true"))
                    dict2 = json.loads(true_answer[i].replace("'", "\"").replace("True", "true"))

                    # Sort list values for stable comparison
                    for d in (dict1, dict2):
                        for key in d:
                            if isinstance(d[key], list):
                                d[key] = sorted(d[key])

                    if dict1 != dict2:
                        print(f"Mismatch at step {i}")
                        dpo_item = get_dpo_data(i + 1, data["output"], response)
                        dpo_item["prompt"] = data["instruction"]
                        preference_data.append(dpo_item)
                        with open(r"preference_data/preference_data.json", 'w', encoding='utf-8') as f:
                            json.dump(preference_data, f, ensure_ascii=False, indent=4)
                        break

                # === Type 3: Nested dictionary comparison ===
                elif i in [4, 5, 6]:
                    dict1 = json.loads(response_answer[i].replace("'", "\"").replace("True", "true"))
                    dict2 = json.loads(true_answer[i].replace("'", "\"").replace("True", "true"))

                    # Custom flexible comparison
                    if not utils.compare_nested_dicts_flexible(dict1, dict2):
                        print(f"Mismatch at step {i}")
                        dpo_item = get_dpo_data(i + 1, data["output"], response)
                        dpo_item["prompt"] = data["instruction"]
                        preference_data.append(dpo_item)
                        with open(r"preference_data/preference_data.json", 'w', encoding='utf-8') as f:
                            json.dump(preference_data, f, ensure_ascii=False, indent=4)
                        break

            break  # Exit retry loop after success

        except Exception as e:
            retry_count += 1
            print(f"Error on index {index}, retry {retry_count}: {e}")
            continue
