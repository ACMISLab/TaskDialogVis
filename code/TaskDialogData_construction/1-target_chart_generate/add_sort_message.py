from llm_enhance.api.deepseek_llm import DeepSeekTextGenerator
import json
import logging
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import json

logger = logging.getLogger('zxc')

SYSTEM_INSTRUCTIONS = """You are an assistant who specializes in writing the perfect Vega-lite visualization tool. Given a Vega-lite visualization configuration and a dataset field, you must understand the dataset and choose a sort type to add sort information to the Vega-lite visualization configuration.\n\n"""


def add_filter(summary: dict, vega, sort_type, text_gen: DeepSeekTextGenerator):
    user_prompt = f"""
# Ability
- Sorting information can only be constructed for nominal and quantitative fields.
- Sorting can only be ascending or descending.
- Make sure the sort information you generate makes sense.
- The data fields selected for sorting in the Sort by a Different Field type are not the same as those used in the original Vega-Lite.

# Dataset Fields
{summary}

# Vega-Lite 
{vega}

# Filter Type
- Sort by a Different Field : To order data by another field, sort can be an encoding sort field definition, such as {{"op": "sum", "field": "xxx", "order":"ascending/descending"}}, {{"op": "mean", "field": "xxx", "order":"ascending/descending"}},{{"op": "count", "field": "xxx", "order":"ascending/descending"}}.
- Sort by Another Encoding : To sort data by another encoding channel, the sort property can be an encoding channel name to sort by (e.g., "x" or "y") with an optional minus prefix for descending sort (e.g., "-x" to sort by x-field, descending).    """
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "assistant",
         "content":
             f"{user_prompt}\n\n You just need to add {sort_type} type sort. You just need to return the json configuration of vega-lite."}]

    response = text_gen.generate_json(messages=messages)

    return response


def generate_sort():
    task_rank = {
        'change_over_time': 3,
        'characterize_distribution': 1,
        'comparison': 1,
        'compute_derived_value': 1,
        'correlate': 1,
        'determine_range': 2,
        'find_anomalies': 3,
        'find_extremum': 1,
        'retrieve_value': 1,
    }
    tasks = list(task_rank.keys())
    for task in tqdm(tasks):

        with open(fr"new_data_chart/{task}-filter.json", 'r',encoding='utf-8') as f:
            init_data = json.load(f)
        textgen = DeepSeekTextGenerator()

        with open(r'C:\Users\admin\PycharmProjects\create_vis_data\get_data_two\data\new_data_summary.json', 'r', encoding='utf-8') as file:
            data_list = json.load(file)
        modify_index = []
        for index, select in enumerate(init_data):
            if select["mark"] != 'bar':
                continue
            print(select["mark"])
            print(index)
            sort_type = "Sort by Another Encoding"

            for data in data_list:
                if data['id'] == select['file']:
                    new_vega = add_filter(data['summary'], select['vega-lite'], sort_type, textgen)
                    new_vega = json.loads(new_vega)
                    init_data[index]['vega-lite'] = new_vega
                    modify_index.append(index)

        with open(rf"new_data_chart/{task}-sort.json", 'w',
                  encoding='utf-8') as f:

            json.dump(init_data, f, ensure_ascii=False, indent=4)
