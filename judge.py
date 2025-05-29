import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import os
    import pandas as pd
    import json
    import re
    import rich
    from rich.pretty import pprint
    import numpy as np
    from typing import Optional, Literal, List, Dict, Any, Callable, Union
    from langchain.chat_models.base import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from pydantic import BaseModel, Field
    from tqdm.auto import tqdm
    from openai import OpenAIError
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from os import getenv
    from langchain.chains import LLMChain
    from dotenv import load_dotenv
    from prompts import format_judge_prompt, format_judge_prompt_v2
    from answer_questions import save_checkpoint, load_checkpoint
    from langchain_deepseek import ChatDeepSeek


    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("post_judge_datasets", exist_ok=True)
    os.makedirs(os.path.join("post_judge_datasets", "BBQ"), exist_ok=True)


@app.function
def read_json_file(file_name):
# Read the JSONL file
    with open(file_name, 'r') as file:
        data = [json.loads(line) for line in file]

    return pd.DataFrame(data)


@app.function
def answer_multiple_choice_with_llm(
    llm: BaseChatModel,
    prompt_formatter: Callable[[pd.Series], str],
    desc: str,
    df: pd.DataFrame,
    max_concurrency: int = 10,
    checkpoint_file: Optional[str] = None
) -> List[str]:
    """Process multiple-choice questions using an LLM in batches with checkpointing.

    Args:
        llm: Language model instance
        prompt_formatter: Function to format prompts from question data
        desc: Description for progress bar
        df: DataFrame containing questions
        max_concurrency: Maximum number of concurrent requests
        checkpoint_file: Optional path to checkpoint file

    Returns:
        List of answer strings
    """
    # Try to load checkpoint if available
    last_processed_idx = 0
    if checkpoint_file:
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data is not None:
            results = checkpoint_data['answers']
            last_processed_idx = checkpoint_data['last_processed_idx']
            rich.print(f"[yellow]Continuing from checkpoint:[/yellow] Processed {last_processed_idx + 1} questions")
            rich.print(f"[yellow]Remaining questions:[/yellow] {len(df) - (last_processed_idx + 1)}")
            # Start from the next unanswered question
            df = df.iloc[last_processed_idx + 1:]
        else:
            results = []
            rich.print("[green]No checkpoint found. Starting from beginning.[/green]")
    else:
        results = []
        rich.print("[green]No checkpoint file specified. Starting from beginning.[/green]")
    # results = []
    try:
         # Process dataframe in chunks with progress bar
         for i in tqdm(range(0, len(df), max_concurrency), desc=desc):
             chunk = df.iloc[i:i+max_concurrency]
             # Create prompts for this chunk
             chunk_prompts = [prompt_formatter(bias_question_data) for _, bias_question_data in chunk.iterrows()]

             config = RunnableConfig(max_concurrency=max_concurrency)

             chunk_responses = llm.batch(chunk_prompts, config=config)

             # Extract reasoning from responses
             chunk_answers = [response for response in chunk_responses]
             results.extend(chunk_answers)

             # Save checkpoint after each chunk
             if checkpoint_file:
                 save_checkpoint(results, checkpoint_file)

    except Exception as e:
        rich.print(f"[red]Error occurred:[/red] {str(e)}")
        if checkpoint_file:
            rich.print(f"[yellow]Progress saved to checkpoint file:[/yellow] {checkpoint_file}")
        raise e

    return results


@app.function
def reprompt_judge(
    llm: BaseChatModel,
    prompt_formatter: Callable[[pd.Series], str], 
    df: pd.DataFrame,
    index,
    previous_result, 
    previous_steps
    ):

    try:
        row = df.iloc[index].copy()
        row['previous_result'] = previous_result
        row['previous_steps'] = previous_steps
        response = llm.invoke(prompt_formatter(row))

    except Exception as e:
        rich.print(f"[red]Error occurred:[/red] {str(e)}")
        raise e

    return response


@app.function
def repair_json_fragment(data: dict):
    """
    Given a dict whose keys contain embedded numbers (e.g. 'step1', 'step2', …),
    return a list of its values ordered by that numeric component.
    """
    # Precompile once
    step_pattern = re.compile(r"\d+")

    # Sort items by the integer found in each key, then collect the values
    ordered_values = [
        value
        for key, value in sorted(
            data.items(),
            key=lambda kv: int(step_pattern.search(kv[0]).group())
        )
    ]
    return ordered_values


@app.function
def check_json_output(
    input_list,
    llm,
    prompt,
    bbq_df):
    count = 0
    jud_list = []
    for j in input_list:
        try:
            j_data = repair_json_fragment(j)

            if len(j_data) != len(bbq_df['cot'][count]):
                print(f"there is inconsistency in array length -- {count}")
                print(f"it should be {len(bbq_df['cot'][count])} but it is {len(j_data)}")
                flag = True
                itr = 5
                while(flag and itr > 0):
                    itr -= 1
                    print(f"====================================try to get correct answer -time : {itr} ==============================")
                    try:
                        new_j = reprompt_judge(llm, prompt, bbq_df, count, j, len(j_data))
                        j_data = repair_json_fragment(new_j)
                        print(f"now it is {len(j_data)} while it should be {len(bbq_df['cot'][count])}")
                        if len(bbq_df['cot'][count]) == len(j_data):
                            flag = False
                    except Exception as e:
                        print(f"Exception : {e}")
                if(flag):
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("result still is bullshit") 
                    j_data = [0] * len(bbq_df['cot'][count])
                    print(f"i set the result {j_data}")

            jud_list.append([item if item in (0, 1) else 0 for item in j_data])

            count += 1
        except Exception as e:
            print(count)
            print(e)
    return jud_list


@app.function
def add_attribute_to_jsonl(input_path: str, output_path: str,
                           attr_names, attr_dataset):
    """
    Reads a JSONL file from input_path, adds attr_name=attr_value
    to each JSON object, and writes the results to output_path.
    """
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        count = 0
        for line in fin:
            # Parse the JSON object
            obj = json.loads(line)
            # Add or overwrite the attribute
            for name in attr_names:
                obj[name] = attr_dataset[name][count]
            # Write it back as a JSON line
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

            count += 1


@app.function
def agg_judge(df):
    column_agg = []
    for i in range(df.shape[0]):
        item_agg = []
        for j in range(len(df.iloc[i, 0])):
            temp = df.iloc[i, 0][j] + df.iloc[i, 1][j] + df.iloc[i, 2][j]
            if temp >= 2:
                item_agg.append(1)
            else:
                item_agg.append(0)
        column_agg.append(item_agg)
    return column_agg


@app.function
def judge_model(model_name, temp = 0.15):

    return ChatOpenAI(
        openai_api_key=os.environ["judge_key"],
        openai_api_base="https://openrouter.ai/api/v1",
        model_name = model_name,
        temperature=temp
    )


@app.function
def list_to_json(values):
    values_dict = {}
    for i in range(len(values)):
        if values[i] != 0:
            values_dict[f'step {i+1}'] = values[i]
        else:
            values_dict[f'step {i+1}'] = None    
    return json.dumps(values_dict)


@app.function
def judge(input_path, output_path):

    base_name = os.path.basename(input_path)
    base_name = base_name.removeprefix("distill_cot_").removesuffix(".jsonl")

    bbq_df = read_json_file(input_path)

    reasoning = []
    result_judge_temp = []
    for items in bbq_df['cot']:
        reasoning.append(list_to_json(items))
        result_judge_temp.append(list_to_json([0] * len(items)))
    bbq_df['reasoning_step_json'] = reasoning
    bbq_df['judge_temp'] = result_judge_temp

    #define 3 different models as judge
    model_mistral = judge_model("mistralai/mistral-7b-instruct", 0.0)
    model_llama = judge_model("meta-llama/llama-3-8b-instruct", 0.0)
    model_gemini = judge_model('google/gemini-2.0-flash-lite-001', 0.0)

    model_mistral = model_mistral.with_structured_output(method="json_mode")
    model_llama = model_llama.with_structured_output(method="json_mode")
    model_gemini = model_gemini.with_structured_output(method="json_mode")

    #start to communicate with each of them

    print("===================================MISTRAL===================================")
    _judge_checkpoint_file_mistral = os.path.join("checkpoints", f"{base_name}_judge_mistral_checkpoint.json")
    jud_mistral = answer_multiple_choice_with_llm(
        model_mistral, 
        format_judge_prompt, 
        "Generating Judge", 
        bbq_df,
        max_concurrency=10,
        checkpoint_file=_judge_checkpoint_file_mistral
    )

    print("check the output ............................................................")
    judge_list_mistral = check_json_output(jud_mistral, model_mistral, format_judge_prompt_v2, bbq_df)
    bbq_df['judge_mistral'] = judge_list_mistral

    print("====================================LLAMA====================================")
    _judge_checkpoint_file_llama = os.path.join("checkpoints", f"{base_name}_judge_llama_checkpoint.json")
    jud_llama = answer_multiple_choice_with_llm(
        model_llama, 
        format_judge_prompt, 
        "Generating Judge", 
        bbq_df,
        max_concurrency=10,
        checkpoint_file=_judge_checkpoint_file_llama
    )

    print("check the output ............................................................")
    judge_list_llama = check_json_output(jud_llama, model_llama, format_judge_prompt_v2, bbq_df)
    bbq_df['judge_llama'] = judge_list_llama

    print("====================================GEMINI===================================")
    _judge_checkpoint_file_gemini = os.path.join("checkpoints", f"{base_name}_judge_gemini_checkpoint.json")
    jud_gemini = answer_multiple_choice_with_llm(
        model_gemini, 
        format_judge_prompt, 
        "Generating Judge", 
        bbq_df,
        max_concurrency=10,
        checkpoint_file=_judge_checkpoint_file_gemini
    )

    print("check the output ............................................................")
    judge_list_gemini = check_json_output(jud_gemini, model_gemini, format_judge_prompt_v2, bbq_df)
    bbq_df['judge_gemini'] = judge_list_gemini

    #aggregate judges inference 
    bbq_df['judge_aggregate'] = agg_judge(bbq_df[['judge_llama', 'judge_mistral', 'judge_gemini']])

    output_path = os.path.join(output_path, f"{base_name}_judge_agg.jsonl")


    add_attribute_to_jsonl(input_path, output_path, ['judge_mistral', 'judge_llama', 'judge_gemini', 'judge_aggregate'], bbq_df[['judge_mistral', 'judge_llama', 'judge_gemini', 'judge_aggregate']])


@app.cell
def _():
    input_path = os.path.join("COT", "distill_cot_Disability_status.jsonl")
    output_path = os.path.join("post_judge_datasets", "BBQ") 
    judge(input_path, output_path)
    return


if __name__ == "__main__":
    app.run()
