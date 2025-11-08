import json
import re
from api.deepseek import DeepSeekTextGenerator
from api.qwen import QwenTextGenerator
from api.local import LocalTextGenerator
from utils.get_dataset_message import get_dataset_message
from api.gpt_oss_120b import GPTOss120BGenerator
from api.llama_70b import Llama70BGenerator
from api.deepseek_r1 import DeepseekR1Generator
from api.openai_4o_mini_api import ChatGPTo4MiniGenerator
import json
from tqdm import tqdm

chart_example_one = """
{
    "Step 1": "Modify Chart",
    "Step 2": {
                "encoding": [
                    "SERVICE"
                ],
                "filter": [
                    "HAAT"
                ]
            },
    "Step 3":["sort"],
    "Step 4":"bar",
    "Step 5":
        {
            "x": {
                "field": "SERVICE"
            },
            "y": {
                "aggregate": "count"
            }
        },
    "Step 6":{
            "gt": [
                "HAAT",
                500
            ]
        },
    "Step 7":{
            "x": {
                "field": "SERVICE",
                "sort": "-y"
            }
        }
}
"""

chart_example_two = """
{
    "Step 1": "Correlate",
    "Step 2": {
                "encoding": [
                    "Total_Revenue_Impact",
                    "Audit_Effectiveness_Score"
                ],
                "filter": [
                    "Year",
                    "Firm_Name"
                ]
            }
        },
    "Step 3":[
                "encoding",
                "mark"
            ],
    "Step 4":"point",
    "Step 5":
        {
            "y": {
                "field": "Total_Revenue_Impact"
            },
            "x": {
                "field": "Audit_Effectiveness_Score"
            }
        },
    "Step 6":{
            "and": [
                {
                    "or": [
                        {
                            "eq": [
                                "Firm_Name",
                                "Deloitte"
                            ]
                        },
                        {
                            "eq": [
                                "Firm_Name",
                                "PwC"
                            ]
                        }
                    ]
                },
                {
                    "gte": [
                        "Year",
                        2020
                    ]
                }
            ]
        },
    "Step 7":{}
}
"""

chart_example_three = """
{
    "Step 1": "Comparison",
    "Step 2": {
            "encoding": [
                "rankingDenominator",
                "type"
            ],
            "filter": []
        },
    "Step 3":["init"],
    "Step 4":"bar",
    "Step 5":
        {
            "x": {
                "field": "type"
            },
            "y": {
                "field": "rankingDenominator",
                "aggregate": "sum"
            }
        },
    "Step 6":{},
    "Step 7":{}
}
"""


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


textgen = DeepseekR1Generator()
# textgen = LocalTextGenerator()
# textgen = QwenTextGenerator()
with open(r"simple_prompt.txt", "r"
        , encoding="utf-8") as file:
    simple_prompt = file.read()

with open(r"../test1.json", "r", encoding="utf-8") as file:
    test_data = file.read()

test_data = json.loads(test_data)
all_of_result = []
for i, dialogue in enumerate(tqdm(test_data)):

    all_of_result.append({"file": dialogue['file'], "dialogues": []})
    dataset = get_dataset_message(dialogue['file'])

    output_template = {
        "analyzing tasks": "<Analysis task of the user’s utterance in the current round of dialogue.>",
        "utterance": "<This round of dialogue the user’s visualization utterance>",
        "chart": "<The Vega-Lite chart inferred to satisfy the user’s utterance intent>"
    }

    history = []
    message = []
    ans = []
    if not filter_dialog_data(dialogue['dialogues']):
        continue
    for index, chart in enumerate(tqdm(dialogue["dialogues"])):
        message = []
        print(i, index)
        if index == 0:

            prompt = simple_prompt.format(
                dataset=dataset,
                pre="None",
                example1=chart_example_one,
                example2=chart_example_two,
                example3=chart_example_three
            )
        else:
            pre = {
                "analyzing tasks": dialogue["dialogues"][index - 1]["analyzing tasks"],
                "utterance": dialogue["dialogues"][index - 1]["utterance"],
                "chart": dialogue["dialogues"][index - 1]["chart"]
            }

            prompt = simple_prompt.format(
                dataset=dataset,
                template=output_template,
                pre=pre,
                example1=chart_example_one,
                example2=chart_example_two,
                example3=chart_example_three
            )

        utterance = chart["utterance"]
        message = [{"role": "user", "content": f"{prompt}+\n user utterance:" + utterance}]
        print(message)
        response = textgen.generate_json(messages=message)
        print(response)
        history.append({"role": "assistant", "content": str(response)})
        all_of_result[-1]["dialogues"].append({"utterance": utterance, "steps": response})
    with open(r"output/deepseek_r1.json", "w", encoding="utf-8") as file:
        json.dump(all_of_result, file, ensure_ascii=False, indent=4)
