import json
import re
import concurrent.futures
import os
from tqdm import tqdm

from api.deepseek_r1_qwen_14b import DeepseekR1Qwen14BGenerator
from api.openai_4o_mini_api import ChatGPTo4MiniGenerator
from data_transform import csv_to_table_schema


def transform_VQL(response):

    pattern = r"<think>.*?</think>"
    cleaned_response = re.sub(pattern, "", response, flags=re.DOTALL)
    start_keyword = "Visualize"
    parts = cleaned_response.split(start_keyword, 1)
    if len(parts) > 1:
        result = (start_keyword + parts[1]).strip()
        return result
    else:
        return None


def filter_dialog_data(dialog):
    for i, chart in enumerate(dialog):
        if chart["chart"]["mark"] not in ["bar", "point", "rect", "line"]:
            return False
        field_type = []
        if "x" in chart["chart"]["encoding"] and "type" in chart["chart"]["encoding"]["x"]:
            if chart["chart"]["encoding"]["x"]["type"] not in field_type:
                field_type.append(chart["chart"]["encoding"]["x"]["type"])
        if "y" in chart["chart"]["encoding"] and "type" in chart["chart"]["encoding"]["y"]:
            if chart["chart"]["encoding"]["y"]["type"] not in field_type:
                field_type.append(chart["chart"]["encoding"]["y"]["type"])
        if "color" in chart["chart"]["encoding"] and "type" in chart["chart"]["encoding"]["color"]:
            if chart["chart"]["encoding"]["color"]["type"] not in field_type:
                field_type.append(chart["chart"]["encoding"]["color"]["type"])
        if "theta" in chart["chart"]["encoding"] and "type" in chart["chart"]["encoding"]["theta"]:
            if chart["chart"]["encoding"]["theta"]["type"] not in field_type:
                field_type.append(chart["chart"]["encoding"]["theta"]["type"])
        if "quantitative" not in field_type:
            return False
    return True

def process_dialogue(dialogue, textgen_instance, prompt_template):

    if not filter_dialog_data(dialogue['dialogues']):
        return {"file": dialogue['file'], "dialogues": []}

    dialogue_result = {"file": dialogue['file'], "dialogues": []}
    dataset = csv_to_table_schema(dialogue['file'])
    print(dataset)
    history_VQL = []

    for index, chart in enumerate(dialogue["dialogues"]):
        for n in range(10):
            if index == 0:
                prompt = prompt_template.format(
                    question=chart["utterance"],
                    dataset=dataset,
                    pre="None",
                )
            else:
                # 依赖上一轮的结果
                pre = history_VQL[-1]
                prompt = prompt_template.format(
                    question=chart["utterance"],
                    dataset=dataset,
                    pre=pre
                )

            message = [{"role": "user", "content": prompt}]

            # 这是主要的耗时操作
            response = textgen_instance.generate_text(messages=message)
            response_VQL = transform_VQL(response)
            print(response_VQL)
            # 如果成功获取VQL，则记录并跳出重试循环
            if response_VQL is not None:
                history_VQL.append(response_VQL)
                dialogue_result["dialogues"].append({"VQL": response_VQL})
                break

    return dialogue_result

def main():

    print("Initializing and loading data...")

    textgen = DeepseekR1Qwen14BGenerator()

    with open(r"prompt.txt", "r", encoding="utf-8") as file:
        simple_prompt = file.read()

    with open(r"../test1.json", "r", encoding="utf-8") as file:
        test_data = json.load(file)


    MAX_WORKERS = 30
    print(f"Starting processing with a thread pool of {MAX_WORKERS} workers...")


    results_in_order = [None] * len(test_data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_to_index = {
            executor.submit(process_dialogue, dialogue, textgen, simple_prompt): i
            for i, dialogue in enumerate(test_data)
        }


        progress_bar = tqdm(concurrent.futures.as_completed(future_to_index), total=len(test_data),
                            desc="Processing Dialogues")

        for future in progress_bar:

            original_index = future_to_index[future]
            try:

                result = future.result()

                results_in_order[original_index] = result
            except Exception as exc:

                dialogue_file = test_data[original_index].get('file', f'unknown file at index {original_index}')
                print(f"An error occurred while processing {dialogue_file}: {exc}")


    final_results = [res for res in results_in_order if res is not None]

    print(f"Processing complete. Found {len(final_results)} valid results.")
    print("Writing results to output.json...")
    with open(r"output.json", "w", encoding="utf-8") as file:
        json.dump(final_results, file, ensure_ascii=False, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()