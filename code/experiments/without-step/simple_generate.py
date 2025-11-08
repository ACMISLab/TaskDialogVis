import json
import re
from api.deepseek import DeepSeekTextGenerator
from api.qwen import QwenTextGenerator
from api.local import LocalTextGenerator
from utils.get_dataset_message import get_dataset_message
import json
from tqdm import tqdm

chart_example_one = """
{
    "transform": [
        {
            "filter": "(datum.start_lat > 30 && datum.start_lat < 40) || (datum.end_lat > 30 && datum.end_lat < 40)"
        }
    ],
    "mark": "bar",
    "encoding": {
        "x": {
            "type": "quantitative",
            "field": "cnt",
            "bin": true
        },
        "y": {
            "type": "quantitative",
            "aggregate": "count"
        },
        "color": {
            "type": "quantitative",
            "field": "airline"
        }
    }
}
"""

chart_example_two = """
{
    "transform": [
        {
            "filter": "(datum.Length > 5 && datum.Width > 3.5) || (datum.Area > 15 && datum.Weight > 40)"
        }
    ],
    "mark": "rect",
    "encoding": {
        "x": {
            "field": "Field",
            "type": "nominal"
        },
        "y": {
            "field": "Genotype",
            "type": "nominal"
        },
        "color": {
            "field": "Weight",
            "type": "quantitative",
            "aggregate": "max"
        }
    }
}
"""

chart_example_three = """
{
    "transform": [
        {
            "filter": "datum.C_score > 7 && datum.Verbal_Reasoning > 6"
        }
    ],
    "mark": "bar",
    "encoding": {
        "y": {
            "type": "quantitative",
            "aggregate": "sum",
            "field": "O_score"
        },
        "x": {
            "type": "nominal",
            "field": "Career",
            "sort": "-y"
        }
    }
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


textgen = DeepSeekTextGenerator()
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
                template=output_template,
                pre="None",
                example1=chart_example_one,
                example2=chart_example_two,
                example3=chart_example_three
            )
        else:
            pre = {
                "analyzing tasks": dialogue["dialogues"][index-1]["analyzing tasks"],
                "utterance": dialogue["dialogues"][index-1]["utterance"],
                "chart": dialogue["dialogues"][index-1]["chart"]
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
        message = [{"role": "user", "content": f"{prompt}+\n this round  user utterance:" + utterance}]
        print(message)
        response = textgen.generate_json_simple(messages=message)
        print(response)
        history.append({"role": "assistant", "content": str(response)})
        all_of_result[-1]["dialogues"].append(response)
    with open(r"output/deepseek_v3_output-5.json", "w", encoding="utf-8") as file:
        json.dump(all_of_result, file, ensure_ascii=False, indent=4)
