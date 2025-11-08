from llm_enhance.api.deepseek_llm import DeepSeekTextGenerator

import json
import logging
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

logger = logging.getLogger('zxc')

SYSTEM_INSTRUCTIONS = """You are an assistant who specializes in writing the perfect Vega-lite visualization tool. 
Given a Vega-lite visualization configuration and a dataset field, you must understand the dataset and choose a 
filter type to add filter information to the Vega-lite visualization configuration.\n\n"""


def add_filter(summary: dict, vega, filter_type, text_gen: DeepSeekTextGenerator):
    user_prompt = f"""
# Ability
- Additional filtering information must be guaranteed to be useful.
- Filter information using no more than three data fields.
- Filtering selected data fields cannot be done using only the data fields in Vega-lite.
- Need to ensure that filtered information makes sense in the context of the dataset.
- The types of symbols in the condition include ! = , = , >= , <= ,>,< in total.

# Dataset Fields
{summary}

# Vega-Lite 
{vega}

# Filter Type
- Single filtering: This filter type contains only one filter condition.
- AND filtering: multiple conditions need to be satisfied at the same time.There are no more than three conditions.
- OR filtering: Only one of the multiple conditions needs to be met. There are no more than three conditions.
- Multi-conditional filtering: Use two or more combinations of conditions to match the conditions, the number of condition combinations should not exceed three. The logical symbols connecting the condition combinations should be different from the logical symbols in the condition combinations. There does not have to be a single data field in a condition combination. The reference cases for condition combinations include AND combinations (datum.xxx === 'xx' || datum.xxx === 'xx'), OR combinations (datum.xxx === 'xx ' && datum.xxx > x), or individual combinations (datum.xxx <= xx).
    """
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "assistant",
         "content":
             f"{user_prompt}\n\n You just need to add {filter_type} type filter transform.filter, transform.filter should be a string. You just need to return the json configuration of vega-lite."}]
    while True:
        try:
            response = text_gen.generate_json(messages=messages)
            break
        except Exception as e:
            pass
    return response

def generate_filter():
    task_rank = {
        'change_over_time': 3,
        'characterize_distribution': 1,
        'comparison': 1,
        'compute_derived_value': 1,
        'correlate': 1,
        'determine_range': 2,
        'find_anomalies': 3,
        'find_extremum': 1,
        'retrieve_value': 1
    }
    tasks = list(task_rank.keys())
    for task in tqdm(tasks):

        with open(fr"C:\Users\admin\PycharmProjects\create_vis_data\get_data_two\init_asp\new_task_chart\{task}.json", 'r',encoding='utf-8') as f:
            init_data = json.load(f)
        textgen = DeepSeekTextGenerator()

        with open(r'C:\Users\admin\PycharmProjects\create_vis_data\get_data_two\data\new_data_summary.json', 'r', encoding='utf-8') as file:
            data_list = json.load(file)
        for index, select in enumerate(init_data):
            print(index)
            random_number = random.random()
            if 0 < random_number <= 0.3:
                filter_type = "Multi-conditional"
            elif 0.3 < random_number <= 0.6:
                filter_type = "Single"
            elif 0.6 < random_number <= 0.8:
                filter_type = "AND"
            else:
                filter_type = "OR"
            for data in data_list:
                if data['id'] == select['file']:
                    new_vega = add_filter(data['summary'], select['vega-lite'], filter_type, textgen)
                    new_vega = json.loads(new_vega)
                    init_data[index]["vega-lite"] = new_vega

            with open(rf"new_data_chart/{task}-filter.json", 'w',
                      encoding='utf-8') as f:

                json.dump(init_data, f, ensure_ascii=False, indent=4)