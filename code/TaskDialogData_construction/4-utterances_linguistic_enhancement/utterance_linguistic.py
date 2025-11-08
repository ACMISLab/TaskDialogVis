import pandas as pd
import os
import json
from tqdm import tqdm
from llm_enhance.api.deepseek_llm import DeepSeekTextGenerator
import random
import time
import random
import re

def get_example(task):

    folderpath = rf'NLV_corpus\{task[0]}'
    output = []

    for index, filename in enumerate(os.listdir(folderpath)):

        if filename.endswith('.csv'):
            filepath = os.path.join(folderpath, filename)
            data = pd.read_csv(filepath)
            tasks = filename.split('.')[0].split('_')
            tworows = data.sample(n=4)
            for num in range(4):
                output.append({"utterance": tworows.iloc[num]["Question"], "visualization task": tasks})

    five_random_values = random.sample(output, 4)
    return five_random_values


textgen = DeepSeekTextGenerator()

# with open(r"final/change_over_time.json", 'r', encoding='utf-8') as file:
#     dialog_data = json.load(file)


with open(r"C:\Users\admin\PycharmProjects\create_vis_data\find_anomalies.json", 'r',
          encoding='utf-8') as file:
    dialog_data = json.load(file)

with open(r"C:\Users\admin\PycharmProjects\create_vis_data\get_data_two\data\summary_deal.json", 'r', encoding='utf-8') as file:
    summaries = json.load(file)

nums = {}
for index, dialog in tqdm(enumerate(dialog_data)):

    summary = ""
    for dataset in summaries:
        if dataset['id'] == dialog['file'].split('.')[0]:
            summary = dataset
            continue
    right = True
    for i, chart in enumerate(dialog['muti-turn']):

        pre_task = ""
        new_task = ""
        pre_vega = ""
        new_vega = ""
        if i == 0:
            pre_task = ["None"]
            new_task = chart["analyzing tasks"]
            pre_vega = "None"
            new_vega = chart["system"]
        else:
            pre_task = dialog['muti-turn'][i - 1]["analyzing tasks"]
            new_task = chart["analyzing tasks"]

            if len(pre_task) > 1:
                dialog['muti-turn'][i - 1]["analyzing tasks"] = [pre_task[0]]
                pre_task = [pre_task[0]]
            if len(new_task) > 1:
                chart["analyzing tasks"] = [new_task[0]]
                new_task = [new_task[0]]

            # pre_vega = {"utterance": dialog['muti-turn'][i - 1]["utterance"],
            #             "vega-lite": dialog['muti-turn'][i - 1]["system"]}
            pre_vega = {"vega-lite": dialog['muti-turn'][i - 1]["system"]}
            new_vega = chart["system"]

            if 'filter' in dialog['muti-turn'][i - 1]["operations"]:
                pre_task.append("Filter")
            if 'sort' in dialog['muti-turn'][i - 1]["operations"]:
                pre_task.append("Sort")
            if 'filter' in chart["operations"]:
                new_task.append("Filter")
            if 'sort' in chart["operations"]:
                new_task.append("Sort")
        print(new_task)
        examples = get_example(new_task)
        options = ["Commands", "Queries", "Questions"]

        # Randomly select one option
        u_type = random.choice(options)
        SYSTEM_PROMPT = f"""
        You are a visualization expert, and your task is to create user utterances for {new_task} analysis tasks. Please, based on the visualization information and Vega-Lite configurations from the previous round of dialogue, as well as the data summary and the Vega-Lite configuration required for constructing the user statement in this round, output a user visualization statement that is contextually relevant and meets the requirements of this round of conversation while satisfying the expression of the analysis task. The generated user statement must meet the expression requirements of the {new_task} analysis task.
        
        # User utterance requirements
        - Ambiguity and unprofessionalism : User utterance must avoid visualization jargon (e.g., “bar chart,” “line chart”), describe requirements only vaguely through natural language (e.g., “compare sales by region “), and may contain ambiguities or unspecified details about fields, chart types, etc.
        - Context relevance: Context relevance requires that the visualization utterance in the current round must be combined with the dialogue history from the previous round (such as mentioned data fields, chart types, filter conditions, sorting conditions) to form a complete visualization requirement.
            - Implicit dependency: The current utterance needs to be based on key information from the previous dialogue (such as field selection, chart type preference, filter conditions, etc.) to clarify the intent. For example:Round 1 (User): “Display sales by region” → The system generates a bar chart;Round 2 (User): “Then compare by month” → This requires reliance on the region and sales fields from round 1 to expand into a faceted bar chart.
            - Implicit association: Users may refer to previous content through pronouns (“this”, “it”) or action words (“again”, “in addition”). For example: “Plot its trend” → “It” refers to the indicator mentioned earlier (such as user growth rate).
        - Avoid subjective descriptions: in the case of filters, users need to directly specify ranges of values, rather than relying on fuzzy terms (e.g., “abnormally high”, “important data”, "low").
        - Disable explicit task expressions: Users should describe their intentions through natural language rather than directly using the names of visualization tasks.
        - Human natural language habits:You must need to generate visualization utterances that conform to human natural language habits. These utterances can be classified into the following three types:
            - Commands: Similar to instructions or requests between people, usually in an imperative tone. For example:
                - "Show me the sales for each region"
                - "Create a bar chart of the number of movies for different years"
                - "Please show me a histogram of weights with 500  intervals."
                - "draw a line chart of daily sales forecasts."
            
            - Queries: Similar to keywords or short web search-like queries, usually concise, and may omit components like predicates. For example:
                - "Cylinders average mpg"
                - "mpg vs displacement > as scatter chart"
                - "average fuel efficiency"
                - "relationship between production budget and worldwide gross"
    
            - Questions: Elicit data visualization needs by asking questions, expecting to see the displayed chart as the answer. For example:
                - "What is our profit based on shipping mode by customer segment?"
                - "How does displacement relate to fuel economy for cars from Europe v. USA?"
                - "What is the average rating for different movie genres?"
                - "How has the weight of cars changed over time?"
        """
        user_prompt = f"""
        # Summary of datasets
        {summary}
        
        # Low-level visualization task
        - Change Over Time: Analyse how the data changes over time series.
        - Characterize Distribution: Characterize the distribution of the data over the set.
        - Comparison: Give emphasis to comparison on different entities.
        - Compute Derived Value: Compute aggregated or binned numeric derived value.
        - Correlate: Determine useful relationships between the columns.
        - Determine Range: Find the span of values within the set.
        - Find Anomalies: Identify any anomalies within the dataset.
        - Find Extremum: Find extreme values of data column.
        - Retrieve Value: Find values of specific columns.
        
        # Previous round of visualization conversations
        {pre_vega}
        
        # This round of conversations Vega-Lite
        {new_vega}
        
        # Output example of {new_task}

        {examples[0]}

        {examples[1]}
        
        {examples[2]}

        {examples[3]}
        
        The syntax of this round of visualization utterance should not be the same as the last round of visualization utterance. 
        Please ensure that you must adhere to the User utterance requirements, Ambiguity and unprofessionalism, Context relevance, Avoid subjective descriptions, Disable explicit task expressions and Human natural language habits. 
        Please return the {u_type} type of the visualization utterance.
        You must refer to the output example to generate this round’s user visualization utterances. 
        User utterances are required to provide accurate chart sort information. 
        User utterances are required to accurately provide aggregation information for encoding channels. 
        User utterances must contain complete filtering information. 
        User utterances must satisfy the expression form of the analysis task.
        Your output should be formatted as a JSON. 
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": user_prompt}]

        response = textgen.generate(messages)
        # time.sleep(5)
        response = response.replace("```json", "").replace("```", "")
        if response.count("}") == 2:
            end_of_first_json = response.find('}') + 1
            response = response[end_of_first_json:]
        try:
            response = json.loads(response)
        except Exception as e:
            try:
                response = textgen.generate(messages)
                response = response.replace("```json", "").replace("```", "")
                if response.count("}") == 2:
                    end_of_first_json = response.find('}') + 1
                    response = response[end_of_first_json:]
                response = json.loads(response)
            except Exception:
                response = textgen.generate(messages)
                response = response.replace("```json", "").replace("```", "")
                if response.count("}") == 2:
                    end_of_first_json = response.find('}') + 1
                    response = response[end_of_first_json:]
                response = json.loads(response)

        if response["utterance"] == "your utterance":
            response = textgen.generate(messages)
            response = response.replace("```json", "").replace("```", "")
            try:
                response = json.loads(response)
            except Exception as e:
                response = textgen.generate(messages)
                response = response.replace("```json", "").replace("```", "")
                response = json.loads(response)
        chart["utterance"] = response["utterance"]

    with open(r"C:\Users\admin\PycharmProjects\create_vis_data\get_data_two\kto\input\find_anomalies.json", 'w',
              encoding='utf-8') as f:
        json.dump(dialog_data, f, ensure_ascii=False, indent=4)
