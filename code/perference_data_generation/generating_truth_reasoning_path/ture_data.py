import json
import re
from tqdm import tqdm
from llm_enhance.api.deepseek_llm import DeepSeekTextGenerator

# Initialize the LLM text generator
text_gen = DeepSeekTextGenerator()


# =========================================================
# Prompt Generation Functions
# =========================================================

def task_prompt_generate(index, chart_answer, field_dict):
    """
    Generate prompt for the analytic task reasoning step.

    Args:
        index (int): The dialogue turn index.
        chart_answer (list): List of previous chart reasoning results.
        field_dict (dict): Dataset field summary.

    Returns:
        list: Chat message for the model prompt.
    """
    with open("prompt/task_prompt.txt", "r") as file:
        prompt_template = file.read()

    # Use the last utterance in the dialogue
    new_chart = {"utterance": chart_answer[-1]["utterance"]}
    result = chart_answer[-1]["analyzing tasks"]

    # Output format template
    output_templates = {
        "Season": "<The reasoning process of the analysis task involved in the user’s utterance. "
                  "The reasoning process should not exceed 100 words.>",
        "Result": result
    }

    # Fill prompt template
    prompt = prompt_template.format(
        data_summary=field_dict,
        new_dialogue=new_chart,
        new_result=result,
        output_templates=output_templates
    )

    messages = [{"role": "assistant", "content": prompt}]
    return messages


def field_prompt_generate(index, chart_answer, field_dict):
    """
    Generate prompt for determining data fields used in visualization.
    """
    with open("prompt/step1prompt.txt", "r") as file:
        step1_template = file.read()

    # Handle the first or subsequent dialogue turn
    if index == 0:
        pre_chart = "None"
        new_chart = {"utterance": chart_answer[-1]["utterance"]}
    else:
        pre_chart = {
            "utterance": chart_answer[-1]["utterance"],
            "fields": chart_answer[-2]["step 1"]
        }
        new_chart = {"utterance": chart_answer[-1]["utterance"]}

    result = chart_answer[-1]["step 1"]
    output_templates = {
        "Season": "<The reasoning process for determining the data fields used in this round of visualization chart. "
                  "The reasoning process should not exceed 100 words.>",
        "Result": result
    }

    # Fill template
    step1_prompt = step1_template.format(
        data_summary=field_dict,
        pre_dialogue=pre_chart,
        new_dialogue=new_chart,
        new_result=result,
        output_templates=output_templates
    )

    messages = [{"role": "assistant", "content": step1_prompt}]
    return messages


def operations_prompt_generate(index, chart_answer, field_dict, history):
    """
    Generate prompt for visual modification operations.
    """
    with open("prompt/step2prompt.txt", "r") as file:
        step2_template = file.read()

    # Define dialogue state
    if index == 0:
        pre_chart = "Since this round is the first round of dialogue for initializing the visualization chart, " \
                    "there is no previous round’s visualization chart."
        new_chart = {
            "utterance": chart_answer[-1]["utterance"],
            "operations": chart_answer[-1]["step 2"]
        }
    else:
        pre_chart = {
            "utterance": chart_answer[-2]["utterance"],
            "operations": chart_answer[-2]["step 2"]
        }
        new_chart = {"utterance": chart_answer[-1]["utterance"]}

    result = chart_answer[-1]["step 2"]
    output_templates = {
        "Season": "<Infer the modification operations applied to the visualization in the current round of dialogue. "
                  "The reasoning process should not exceed 100 words.>",
        "Result": result
    }

    step2_prompt = step2_template.format(
        data_summary=field_dict,
        pre_steps=history,
        pre_dialogue=pre_chart,
        new_dialogue=new_chart,
        new_result=result,
        output_templates=output_templates
    )

    messages = [{"role": "assistant", "content": step2_prompt}]
    return messages


def mark_prompt_generate(index, chart_answer, field_dict, history):
    """
    Generate prompt for determining visualization mark type.
    """
    with open("prompt/mark_prompt.txt", "r") as file:
        mark_template = file.read()

    # Handle the first or subsequent round
    if index == 0:
        pre_chart = "Since this round is the first round of dialogue for initializing the visualization chart, " \
                    "there is no previous round’s visualization chart."
        new_chart = {
            "utterance": chart_answer[-1]["utterance"],
            "analyzing task": chart_answer[-1]["analyzing tasks"]
        }
    else:
        pre_chart = {
            "utterance": chart_answer[-2]["utterance"],
            "analyzing task": chart_answer[-2]["analyzing tasks"],
            "mark": chart_answer[-2]["step 3"]
        }
        new_chart = {
            "utterance": chart_answer[-1]["utterance"],
            "analyzing task": chart_answer[-1]["analyzing tasks"]
        }

    result = chart_answer[-1]["step 3"]
    output_templates = {
        "Season": "<The reasoning process for determining the visualization type in the current round of dialogue. "
                  "The reasoning process should not exceed 100 words.>",
        "Result": result
    }

    mark_prompt = mark_template.format(
        data_summary=field_dict,
        pre_dialogue=pre_chart,
        pre_steps=history,
        new_dialogue=new_chart,
        new_result=result,
        output_templates=output_templates
    )

    messages = [{"role": "assistant", "content": mark_prompt}]
    return messages


def channel_prompt_generate(index, chart_answer, field_dict, history):
    """
    Generate prompt for visualization channel encoding reasoning.
    """
    with open("prompt/channel_prompt.txt", "r") as file:
        channel_template = file.read()

    if index == 0:
        pre_chart = "Since this round is the first round of dialogue for initializing the visualization chart, " \
                    "there is no previous round’s visualization chart."
        new_chart = {"utterance": chart_answer[-1]["utterance"], "analyzing task": "init"}
    else:
        pre_chart = {
            "utterance": chart_answer[-2]["utterance"],
            "analyzing task": chart_answer[-2]["analyzing tasks"],
            "channel": chart_answer[-2]["step 5"]
        }
        new_chart = {
            "utterance": chart_answer[-1]["utterance"],
            "analyzing task": chart_answer[-1]["analyzing tasks"]
        }

    result = chart_answer[-1]["step 5"]
    output_templates = {
        "Season": "<The reasoning process for mapping visualization channels to data fields in this round of conversation. "
                  "The reasoning process should not exceed 100 words.>",
        "Result": result
    }

    channel_prompt = channel_template.format(
        data_summary=field_dict,
        pre_steps=history,
        pre_dialogue=pre_chart,
        new_dialogue=new_chart,
        new_result=result,
        output_templates=output_templates
    )

    messages = [{"role": "assistant", "content": channel_prompt}]
    return messages


def filter_prompt_generate(index, chart_answer, field_dict, history):
    """
    Generate prompt for filtering reasoning step.
    """
    with open("prompt/filter_prompt.txt", "r") as file:
        filter_template = file.read()

    if index == 0:
        pre_chart = "Since this round is the first round of dialogue for initializing the visualization chart, " \
                    "there is no previous round’s visualization chart."
        new_chart = {"utterance": chart_answer[-1]["utterance"], "field": chart_answer[-1]["step 1"]}
    else:
        pre_chart = {
            "utterance": chart_answer[-2]["utterance"],
            "field": chart_answer[-2]["step 1"],
            "filter": chart_answer[-2]["step 6"]
        }
        new_chart = {"utterance": chart_answer[-1]["utterance"], "field": chart_answer[-1]["step 1"]}

    result = chart_answer[-1]["step 6"]
    output_templates = {
        "Season": "<The reasoning process for filtering information in the current round's visualization chart. "
                  "The reasoning process should not exceed 100 words.>",
        "Result": result
    }

    filter_prompt = filter_template.format(
        data_summary=field_dict,
        pre_steps=history,
        pre_dialogue=pre_chart,
        new_dialogue=new_chart,
        new_result=result,
        output_templates=output_templates
    )

    messages = [{"role": "assistant", "content": filter_prompt}]
    return messages


def sort_prompt_generate(index, chart_answer, field_dict, history):
    """
    Generate prompt for sorting reasoning step.
    """
    with open("prompt/sort_prompt.txt", "r") as file:
        sort_template = file.read()

    if index == 0:
        pre_chart = "Since this round is the first round of dialogue for initializing the visualization chart, " \
                    "there is no previous round’s visualization chart."
        new_chart = {"utterance": chart_answer[-1]["utterance"], "channel": chart_answer[-1]["step 5"]}
    else:
        pre_chart = {
            "utterance": chart_answer[-2]["utterance"],
            "channel": chart_answer[-2]["step 5"],
            "sort": chart_answer[-2]["step 7"]
        }
        new_chart = {"utterance": chart_answer[-1]["utterance"], "channel": chart_answer[-1]["step 5"]}

    result = chart_answer[-1]["step 7"]
    output_templates = {
        "Season": "<The reasoning process for sorting information in the current round of visualization chart. "
                  "The reasoning process should not exceed 100 words.>",
        "Result": result
    }

    sort_prompt = sort_template.format(
        data_summary=field_dict,
        pre_steps=history,
        pre_dialogue=pre_chart,
        new_dialogue=new_chart,
        new_result=result,
        output_templates=output_templates
    )

    messages = [{"role": "assistant", "content": sort_prompt}]
    return messages


import json
import re
from tqdm import tqdm
from llm_enhance.api.deepseek_llm import DeepSeekTextGenerator

text_gen = DeepSeekTextGenerator()


# =========================================================
# Utility functions
# =========================================================

def extract_reasoning_and_result(text):
    """
    Extract reasoning ('Season') and final result ('Result') parts from model output.
    Returns a tuple (season, result). Handles common variations in formatting.
    """
    try:
        season_match = re.search(r"Season\s*[:：]\s*(.*?)(?=Result\s*[:：]|$)", text, re.S | re.I)
        result_match = re.search(r"Result\s*[:：]\s*(.*)", text, re.S | re.I)
        season = season_match.group(1).strip() if season_match else ""
        result = result_match.group(1).strip() if result_match else ""
        return season, result
    except Exception:
        return "", text.strip()


def run_with_retry(prompt_func, *args, max_retries=2, **kwargs):
    """
    Run a single reasoning step with automatic retry on failure.
    """
    for attempt in range(max_retries):
        try:
            messages = prompt_func(*args, **kwargs)
            response = text_gen.single_turn_generate(messages)
            season, result = extract_reasoning_and_result(response)
            if result:
                return season, result
        except Exception as e:
            print(f"[Retry {attempt+1}] Error in {prompt_func.__name__}: {e}")
    return "", ""


# =========================================================
# Main dialogue processing logic
# =========================================================

def turn_data_generate(dialog_data, field_dict, save_path="data/step_reasoning.jsonl"):
    """
    Process a single multi-turn dialogue and generate reasoning chains step by step.

    Args:
        dialog_data (dict): One dialogue sample with chart/utterance pairs.
        field_dict (dict): Dataset field summary.
        save_path (str): Output path to save incremental results.

    Returns:
        dict: Completed dialogue reasoning record.
    """

    dialogue_result = {"dialog_id": dialog_data["dialog_id"], "turns": []}

    # List of all reasoning steps to execute sequentially
    steps = [
        ("analyzing tasks", task_prompt_generate),
        ("step 1", field_prompt_generate),
        ("step 2", operations_prompt_generate),
        ("step 3", mark_prompt_generate),
        ("step 5", channel_prompt_generate),
        ("step 6", filter_prompt_generate),
        ("step 7", sort_prompt_generate),
    ]

    chart_answers = []
    history = []  # accumulate previous step summaries for context

    for turn_idx, utterance_data in enumerate(tqdm(dialog_data["dialog"])):
        turn_record = {"utterance": utterance_data["utterance"]}

        # Sequentially execute each reasoning step
        for step_name, prompt_func in steps:
            season, result = run_with_retry(
                prompt_func, turn_idx, chart_answers, field_dict, history=history
            )
            turn_record[step_name] = result
            if season:
                history.append(f"{step_name}: {season}")

        chart_answers.append(turn_record)
        dialogue_result["turns"].append(turn_record)

        # Incremental save (prevents total data loss on interrupt)
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn_record, ensure_ascii=False) + "\n")

    return dialogue_result


# =========================================================
# Example: Batch Processing Wrapper
# =========================================================

def process_dataset(dataset_path, field_dict_path, save_path="data/step_reasoning.jsonl"):
    """
    Process all dialogues in a dataset and save reasoning data for each.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(field_dict_path, "r", encoding="utf-8") as f:
        field_dict = json.load(f)

    all_results = []
    for dialog_data in tqdm(dataset, desc="Processing dialogues"):
        result = turn_data_generate(dialog_data, field_dict, save_path=save_path)
        all_results.append(result)

    print(f"\n✅ Completed all dialogues. Results saved to {save_path}")
    return all_results


