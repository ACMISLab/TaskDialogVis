from llm_enhance.api.deepseek_llm import DeepSeekTextGenerator
import json
import logging
import os
import pandas as pd
import random
from tqdm import tqdm

logger = logging.getLogger('zxc')

SYSTEM_INSTRUCTIONS = """
As a user focused on data visualization applications, I need you to automatically generate Visualization utterance in natural language based on the provided data summary and analysis tasks and its Vega-Lite.

# Analysis Tasks type
- Retrieve Value: Extract values from a specific column.
- Find Extremum: Identify extreme values in data attributes.
- Find Anomalies: Detect anomalies in the dataset.
- Determine Range: Identify the range of values within a set.
- Compute Derived Value: Calculate aggregated or binned values.
- Comparison: Highlight comparisons between different entities.
- Correlate: Explore relationships between entities.
- Characterize Distribution: Describe data distribution patterns.
- Change Over Time: Analyze temporal changes.

# Information that a Visualization Utterance May Contain

The utterances you generate should completely cover all conditional filtering and sorting information defined in the Vega-Lite configuration. These utterances may contain information on the following four aspects, used to describe various aspects of data visualization:

-   Attributes: Words in the query utterance that map to data attributes. Attribute references can be further divided into four subcategories:
    -   Explicit: Words in the query directly correspond to a part of the data attribute (e.g., "mpg" -> MPG, "genre" -> Major Genre, "sales" -> Weekly_Sales).
    -   Semantic: Words in the query are synonyms or semantically similar to the dataset attributes (e.g., "heavy" -> Weight, "fuel economy" -> MPG, "over time" -> Year, "temperature" -> Temperature).
    -   Value-based: Words in the query refer to cell values instead of column names (e.g., "1995 to 2010" -> Release Year, "furniture, office supplies, and technology" -> Category, "below 7" -> Unemployment < 7).
    -   Implicit: Attributes are requested indirectly through the visualization type (e.g., if there is only one temporal attribute, requesting a line chart implies a reference to that attribute).

-   Chart Types: Words or phrases in the query utterance that map to specific visualization types. Chart type references can be further divided into two subcategories:
    -  Explicit: Words or phrases in the query directly specify the desired chart type (e.g., "bar chart" -> Bar Chart, "scatterplot" -> Scatterplot, "line graph" -> Line Graph, "histogram" -> Histogram, "pie chart" -> Pie Chart).
    -  Implicit: The query does not explicitly state the chart type, but the desired chart type can be inferred from other information in the query. (e.g., "Show me the trend of sales over time"  -> Line Chart, "What is the distribution of student heights?"-> Histogram, "Compare the sales performance of different product categories" -> Bar Chart, "Show me the proportion of each category in total sales" -> Pie Chart, "Visualize the relationship between sales and profit for each region" -> Grouped Bar Chart or Scatterplot)

-   Encoding: Words or phrases in the query utterance that specify or imply how data attributes should be visually represented in the chart. Encoding references can be further divided into two subcategories:
    - Explicit: Words in the query directly specify the visual encoding channel and the attribute it should be mapped to (e.g., "color by origin", "x-axis is MPG", "size by sales", "weight on the X Axis"). These typically involve keywords like "by", "on the", "as the", combined with visual channel names like "color", "x-axis", "y-axis", "shape", "size", "facet".
    - Implicit: The visual encoding is not directly stated but is implied through the chart type, context, or conventional visualization practices. The system needs to infer the intended encoding based on these cues.
    - Chart Type Implied: The user specifies a chart type that has conventional encodings associated with it (e.g., "scatterplot of MPG and displacement" implies mapping one attribute to the x-axis and the other to the y-axis, but doesn't specify which is which).
    - Context/Convention Implied: The user mentions attributes or a general task, and the system infers the encoding based on common practices, data types, or previous interactions (e.g., "Visualize profit across states" in the context of a bar chart might imply mapping states to the x-axis (categorical) and profit to the y-axis (height); "Show the distribution of sales" might imply a histogram where sales is on the x-axis and the count of sales in each bin is on the y-axis (height)).
-   Aggregation: Includes one or more words that map to the type of mathematical transformation required to create the chart (e.g., sum, average, count). Aggregation references can be further subdivided into:
    -   Explicit: For example, "total gross" -> SUM(Worldwide Gross), "number of orders" -> COUNT, "total sales" -> sum(Weekly_Sales).
    -   Implicit: Inferred through the requested chart type or wording, for example, "histogram" or "How many" -> COUNT, "stacked bar chart" -> SUM or COUNT.
```
"""


def init_generate(summary, text_gen, analyze_task, goal):
    global field_type, init_task, field

    if "x" in goal["encoding"] and "field" in goal["encoding"]["x"] and goal["encoding"]['x']["type"] != "temporal":
        field = goal["encoding"]["x"]["field"]
        field_type = goal["encoding"]['x']["type"]
    elif "y" in goal["encoding"] and "field" in goal["encoding"]["y"]:
        field = goal["encoding"]["y"]["field"]
        field_type = goal["encoding"]['y']["type"]
    elif "color" in goal["encoding"] and "field" in goal["encoding"]["color"]:
        field = goal["encoding"]["color"]["field"]
        field_type = goal["encoding"]['color']["type"]

    analyze_task = ' '.join(word.capitalize() for word in analyze_task.split('_'))

    task_list = ["Retrieve Value", "Find Extremum", "Find Anomalies",
                 "Determine Range", "Comparison", "Correlate", "Characterize Distribution","Compute Derived Value","Change Over Time"]
    task_list.remove(analyze_task)

    if field_type == "quantitative":
        while "Compute Derived Value" in task_list:
            task_list.remove("Compute Derived Value")
        init_task = random.choice(task_list)
    elif field_type == "temporal":
        init_task = random.choice(["Change Over Time"])
    elif field_type in ["ordinal", "nominal"]:
        for element in ["Change Over Time", "Correlate", "Change Over Time"]:
            while element in task_list:
                task_list.remove(element)
        init_task = random.choice(task_list)

    task_chart_type = {
        'Retrieve Value': 'rect',
        'Find Extremum': 'bar or point',
        'Find Anomalies': 'boxplot or point',
        'Determine Range': 'boxplot',
        'Compute Derived Value': 'rect or arc or bar must aggregate',
        'Comparison': 'line or point or bar',
        'Correlate': 'point',
        'Characterize Distribution': 'boxplot or histogram or arc',
        'Change Over Time': 'line or area'
    }

    chart_type = task_chart_type[init_task]

    user_prompt = f"""
    # Data Summary
    ```json
    {summary}
    ```
    # Output Template 
    {{
      "Task": "",
      "Utterance": "",
      "Vega-Lite":""
    }}
    
Please generate a Vega-Lite visualization configurations for visual utterance that are simply {init_task} and contain {field} fields. Only two data fields can be used. Only mark, encoding information is needed. Only  {chart_type} chart types can be used. Only two data fields can be used. Only the x and y axes can be used. A chart of type rect must have x,y,color. No need for title. The Utterance must implicitly satisfy the intent of the type of task being analyzed. 
"""

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "assistant",
         "content": user_prompt}]

    response = text_gen.generate_json(messages=messages)
    return response
