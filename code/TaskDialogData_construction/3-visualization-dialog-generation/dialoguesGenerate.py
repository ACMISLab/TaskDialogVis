import json
import logging
import os
import random
import pandas as pd
from tqdm import tqdm
from json import JSONDecodeError
from llm_enhance.api.deepseek_llm import DeepSeekTextGenerator

# Initialize logger
logger = logging.getLogger('zxc')
logger.setLevel(logging.INFO)

# Optional: set up console handler if not configured globally
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def conversion_standard(json_file, textgen):
    """
    Convert a given JSON-like string into a standard JSON format using the LLM.
    Includes error handling and logging for decoding issues.
    """
    messages = [
        {"role": "system", "content": "Please convert the provided json string to standard json format for output."},
        {"role": "assistant", "content": json_file}
    ]

    try:
        response = textgen.generate_json(messages=messages)
        return response
    except JSONDecodeError as e:
        logger.error(f"JSON decoding failed during conversion_standard: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in conversion_standard: {e}")

    # Return a safe default in case of failure
    return {"error": "conversion_failed"}


def user_agent(textgen, summary, pre, task, target):
    """
    Simulate a user utterance for a given visualization scenario with safe fallback.
    """
    output_template = {
        "analytic task": "<Analysis task of the user's utterance.>",
        "utterance": "<This round of dialogue: the user's visualization utterance>"
    }

    try:
        with open(r"user_prompt.txt", "r", encoding="utf-8") as file:
            user_prompt = file.read()
    except FileNotFoundError:
        logger.error("Missing 'user_prompt.txt' file.")
        return {"error": "missing_user_prompt"}

    prompt = user_prompt.format(
        summary=summary,
        pre=pre,
        task=task,
        target=target,
        template=output_template
    )

    try:
        message = [{"role": "user", "content": prompt}]
        response = textgen.generate_json(messages=message)
        return response
    except Exception as e:
        logger.error(f"Error in user_agent: {e}")
        return {"utterance": "<Error generating user utterance>", "analytic task": None}


def expert_agent(textgen, summary, utterance, pre, reason, origin_chart):
    """
    Expert agent that interprets user utterances and generates corresponding Vega-Lite chart specifications.
    Includes retry mechanism and detailed logging for debugging.
    """
    # Example charts for few-shot prompting
    example_one = """
    {
        "utterance": "What is the count of properties listed for each type?",
        "analytic task": "Compute Derived Value",
        "chart": {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"name": "vegalitedata"},
            "mark": "bar",
            "encoding": {
                "x": {"field": "TYPE", "type": "nominal"},
                "y": {"aggregate": "count", "type": "quantitative"}
            }
        }
    }
    """

    example_two = """
    {
        "utterance": "Narrow it down to Electronics and Home Appliances categories.",
        "analytic task": "Modify Chart",
        "chart": {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": "vegalitedata",
            "mark": "rect",
            "encoding": {
                "x": {"field": "Category", "type": "nominal"},
                "y": {"field": "Status", "type": "nominal"},
                "color": {"field": "Price", "type": "quantitative", "aggregate": "mean"}
            },
            "transform": [
                {"filter": "datum.Category === 'Electronics' || datum.Category === 'Home Appliances'"}
            ]
        }
    }
    """

    example_three = """
    {
        "utterance": "Among books published after 1990 with ratings above 4, which books have the highest ratings and revenue for publishers?",
        "analytic task": "Find Extremum",
        "chart": {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"name": "vegalitedata"},
            "transform": [
                {"filter": "datum.Publishing_Year > 1990 && datum.Book_average_rating > 4"}
            ],
            "mark": "point",
            "encoding": {
                "x": {"field": "Book_average_rating", "type": "quantitative"},
                "y": {"field": "publisher_revenue", "type": "quantitative"}
            }
        }
    }
    """

    output_template = {
        "utterance": "<This round of dialogue: user's visualization utterance>",
        "analytic task": "<Analysis task of the user's utterance>",
        "chart": "<The Vega-Lite chart inferred to satisfy the user's intent>"
    }

    try:
        with open(r"expert_prompt.txt", "r", encoding="utf-8") as file:
            expert_prompt = file.read()
        with open(r"expert_with_reason.txt", "r", encoding="utf-8") as file:
            expert_with_prompt = file.read()
    except FileNotFoundError as e:
        logger.error(f"Missing prompt file in expert_agent: {e}")
        return {"error": "missing_expert_prompt"}

    # Build prompt based on whether evaluator feedback exists
    try:
        if reason is None:
            prompt = expert_prompt.format(
                summary=summary,
                pre=pre,
                utterance=utterance,
                template=output_template,
                example1=example_one,
                example2=example_two,
                example3=example_three
            )
        else:
            prompt = expert_with_prompt.format(
                summary=summary,
                suggestion=reason,
                origin_chart=origin_chart,
                utterance=utterance,
                template=output_template,
                example1=example_one,
                example2=example_two,
                example3=example_three
            )

        message = [{"role": "user", "content": prompt}]
        response = textgen.generate_json(messages=message)
        return response

    except JSONDecodeError as e:
        logger.warning(f"JSON decoding failed in expert_agent: {e}")
    except Exception as e:
        logger.error(f"Error in expert_agent: {e}")

    return {"chart": None, "utterance": utterance, "analytic task": None}


def evaluator_agent(textgen, summary, chart, utterance):
    """
    Evaluator agent that critiques the generated Vega-Lite chart and suggests improvements.
    Includes protection against invalid LLM responses.
    """
    output_template = {
        "utterance": "<This round of dialogue: user's visualization utterance>",
        "suggest": "<Suggested changes to the Vega-Lite visualization specification.>"
    }

    try:
        with open(r"expert_prompt.txt", "r", encoding="utf-8") as file:
            expert_prompt = file.read()
    except FileNotFoundError:
        logger.error("Missing 'expert_prompt.txt' file for evaluator_agent.")
        return {"error": "missing_evaluator_prompt"}

    prompt = expert_prompt.format(
        summary=summary,
        chart=chart,
        utterance=utterance,
        template=output_template
    )

    try:
        message = [{"role": "user", "content": prompt}]
        response = textgen.generate_json(messages=message)
        return response
    except Exception as e:
        logger.error(f"Error in evaluator_agent: {e}")
        return {"reason": "<Evaluator failed to generate suggestion>"}


def muti_generate(target_chart, init_vega, summary, init_task, init_utterance, target_task, text_gen):
    """
    Multi-turn dialogue simulation pipeline that iteratively generates user and expert exchanges.
    Includes fail-safe logic to prevent crashes in case of intermediate agent failure.
    """
    output = {
        "dialogues": [
            {
                "turns": 1,
                "analytic task": [init_task],
                "utterance": init_utterance,
                "chart": init_vega
            }
        ]
    }

    for index in range(5):
        try:
            # Stage 1: Initial analytic task rounds
            if index <= 2:
                user_response = user_agent(text_gen, summary, output["dialogues"][-1]["chart"], init_task, target_chart)
                reason, origin_chart, expert_response = None, None, None

                for _ in range(3):
                    expert_response = expert_agent(text_gen, summary, user_response["utterance"], init_vega, reason, origin_chart)
                    origin_chart = expert_response.get("chart")
                    evaluator_response = evaluator_agent(text_gen, summary, origin_chart, user_response["utterance"])
                    reason = evaluator_response.get("reason")

                output["dialogues"].append({
                    "turns": index + 2,
                    "analytic task": [init_task],
                    "utterance": user_response.get("utterance"),
                    "chart": expert_response.get("chart")
                })

            # Stage 2: Transition to target analytic task
            else:
                user_response = user_agent(text_gen, summary, output["dialogues"][-1]["chart"], target_task, target_chart)
                reason, origin_chart, expert_response = None, None, None

                for _ in range(3):
                    expert_response = expert_agent(text_gen, summary, user_response["utterance"], init_vega, reason, origin_chart)
                    origin_chart = expert_response.get("chart")
                    evaluator_response = evaluator_agent(text_gen, summary, origin_chart, user_response["utterance"])
                    reason = evaluator_response.get("reason")

                output["dialogues"].append({
                    "turns": index + 2,
                    "analytic task": [target_task],
                    "utterance": user_response.get("utterance"),
                    "chart": expert_response.get("chart")
                })

        except Exception as e:
            logger.error(f"Error during multi-turn generation at step {index + 1}: {e}")

    return output
