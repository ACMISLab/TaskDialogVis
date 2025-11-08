# TaskDialogVis
#### NEWS
- 🔥 Data Repo in Hugging Face :hugs: : [TaskDialogData](https://huggingface.co/datasets/GZUzxc/TaskDialogData)
- 🔥 Model in Hugging Face :hugs: : [TaskDialogVis](https://huggingface.co/GZUzxc/TaskDialogVis_Model)

## Introduction

In real-world data analysis scenarios, users typically refine their goals through **multi-turn interactions**. Directly mapping single-turn natural language to visualization specifications suffers from **context forgetting** and **parameter hallucination**.
**TaskDialogViz** addresses these issues by inserting an intermediate **analytic task layer** between natural language and visualization specs, breaking generation into a **stepwise reasoning chain**, and applying **Step-DPO** to reduce error propagation across steps. The dataset construction, model pipeline, and experiments are detailed in the paper.

## Key Contributions

**Task formulation:** Introduce ATRCovis (Analytic Task Reasoning for Conversational Text-to-Visualization) by using an explicit analytic-task representation to support multi-turn conversational visualization.

**Dataset:** Provide **TaskDialogData** — multi-turn dialogues annotated with analytic tasks and corresponding Vega-Lite specifications (e.g., ~109 tables, 748 dialogues, 3490 charts in the dataset as reported).

**Method:** Propose **TaskDialogViz**, which combines (A) a decomposed stepwise reasoning chain (task recognition, field extraction, change detection, chart type/encoding/filter/sort generation) and (B) Step-DPO for stepwise preference optimization to improve intermediate output correctness.

**Evaluation:** Extensive comparisons (rule-based, prompting, CoT, etc.), ablations, and user studies demonstrating improvements in task consistency and chart consistency.

## Code Structure

The repository is organized as follows:

```
code_and_data/
├── code/
│   ├── TaskDialogData_construction/
│   ├── perference_data_generation/
│   ├── step-dpo/
│   └── experiments/
└── data/
```

Here is a detailed breakdown of each directory:

*   **`data/`**: This directory contains the final `TaskDialogData` dataset used for training and evaluation.

*   **`code/`**: This directory houses all the source code, structured to reflect the key stages of our methodology.
    *   **`TaskDialogData_construction/`**: Contains the scripts for our four-step, multi-agent pipeline to construct the `TaskDialogData` dataset. If you want to understand how the data was created, start here.
    *   **`perference_data_generation/`**: Includes the scripts for generating the stepwise preference data required for Step-DPO. This process creates the "winning" and "losing" reasoning steps that are crucial for training.
    *   **`step-dpo/`**: The core implementation of our `TaskDialogViz` model. This directory contains the code for training the model using the Stepwise Preference Optimization (Step-DPO) algorithm.
    *   **`experiments/`**: Contains all scripts needed to run evaluations, perform ablation studies, and reproduce the results presented in our paper. Use these scripts to compare `TaskDialogViz` against the baselines.

## How *TaskDialogData* was Created

The TaskDialogData dataset was constructed using a sophisticated four-step, LLM-driven pipeline to ensure each dialogue is logically coherent, contextually relevant, and linguistically natural.

1. **Target Visualization Generation**: We first generate the final, complex visualization (V_target) that serves as the end goal of a dialogue. An Answer Set Programming (ASP) solver creates a core chart, which is then enriched by an LLM with advanced features like sorting and filtering.
2. **Seed Visualization Generation**: A simpler, foundational visualization (V_seed) is created to act as the conversation's starting point. This seed is intentionally distinct from the target to necessitate a multi-turn, exploratory interaction.
3. **Visualization Dialogue Generation**: A multi-agent system simulates the conversation to bridge the gap between the seed and target visualizations. This involves:
   - A **User Agent** to generate natural language requests.
   - A **Visualization Expert Agent** to produce the corresponding Vega-Lite specifications.
   - An **Evaluator Agent** to verify correctness and provide feedback, ensuring a logically sound dialogue.
4. **Utterance Linguistic Enhancement**: Finally, the programmatic utterances from the User Agent are refined. An LLM rewrites them using the full conversational context, transforming them into more natural, human-like expressions while preserving the original analytical intent.

## How *TaskDialogViz* is Trained

The training process for TaskDialogViz is designed to master conversational visualization by combining a structured reasoning framework with a granular optimization technique. This approach effectively mitigates common LLM issues like context forgetting and hallucination.

The core of our method involves two key components:

1. **Stepwise Reasoning Chain**: We decompose the complex task of generating a visualization into a sequence of smaller, logical sub-tasks (e.g., *Analytic Task Identification*, *Data Field Extraction*, *Chart Parameter Generation*). This transforms the opaque generation process into a transparent and controllable workflow.
2. **Stepwise Preference Optimization (Step-DPO)**: To ensure high accuracy at each stage, we use Step-DPO. Unlike traditional methods that only optimize the final output, Step-DPO trains the model to distinguish between correct and incorrect reasoning paths at **each intermediate step**. This is crucial for preventing early errors from derailing the entire generation process.

The training pipeline consists of the following stages:

1. **Cold Start (SFT)**: We begin by performing Supervised Fine-Tuning (SFT) on a base LLM. This teaches the model to follow the structured, multi-step reasoning format, creating a stable reference model.
2. **Preference Data Generation**:
   - We first generate the ground-truth ("winning") reasoning paths for our training data.
   - Next, we use the reference model to generate candidate ("losing") reasoning paths, which naturally contain various errors.
   - By comparing these paths step-by-step, we identify the first point of divergence and create a preference tuple: (prompt, correct_step, incorrect_step).
3. **Step-DPO Training**: The model is then trained on these preference tuples using the Step-DPO objective. This teaches the model to prefer the correct reasoning choice at every stage, significantly enhancing its accuracy and reliability.

This two-pronged approach ensures that TaskDialogViz not only understands the user's intent but also follows a robust and logical path to generate the correct visualization.

## Data Usage

see `./data/` dir
