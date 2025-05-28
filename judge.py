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


@app.function
def read_json_file(file_name):
# Read the JSONL file
# with open('distill_cot_Age_cleaned.jsonl', 'r') as file:
# with open('reasoning_steps.jsonl', 'r') as file:
    with open(file_name, 'r') as file:
        data = [json.loads(line) for line in file]

    return pd.DataFrame(data)


@app.function
def answer_multiple_choice_with_llm(
    llm: BaseChatModel,
    prompt_formatter: Callable[[pd.Series], str],
    desc: str,
    df: pd.DataFrame,
    max_concurrency: int = 20
    # checkpoint_file: Optional[str] = None
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
    # # Try to load checkpoint if available
    # if checkpoint_file:
    #     checkpoint_data = load_checkpoint(checkpoint_file)
    #     if checkpoint_data is not None:
    #         results = checkpoint_data['answers']
    #         last_processed_idx = checkpoint_data['last_processed_idx']
    #         rich.print(f"[yellow]Continuing from checkpoint:[/yellow] Processed {last_processed_idx + 1} questions")
    #         rich.print(f"[yellow]Remaining questions:[/yellow] {len(df) - (last_processed_idx + 1)}")
    #         # Start from the next unanswered question
    #         df = df.iloc[last_processed_idx + 1:]
    #     else:
    #         results = []
    #         rich.print("[green]No checkpoint found. Starting from beginning.[/green]")
    # else:
    #     results = []
    #     rich.print("[green]No checkpoint file specified. Starting from beginning.[/green]")
    results = []
    try:
         # Process dataframe in chunks with progress bar
         for i in tqdm(range(0, len(df), max_concurrency), desc=desc):
             chunk = df.iloc[i:i+max_concurrency]
             # Create prompts for this chunk
             chunk_prompts = [prompt_formatter(bias_question_data) for _, bias_question_data in chunk.iterrows()]
             # save_checkpoint = chunk_prompts
             # Process this chunk

             config = RunnableConfig(max_concurrency=max_concurrency)
             chunk_responses = llm.batch(chunk_prompts, config=config)
             # print(chunk_responses)
             # break
             # Extract reasoning from responses
             chunk_answers = [response.content for response in chunk_responses]
             results.extend(chunk_answers)

             # Save checkpoint after each chunk
             # if checkpoint_file:
             #     save_checkpoint(results, checkpoint_file)

    except Exception as e:
        rich.print(f"[red]Error occurred:[/red] {str(e)}")
        # if checkpoint_file:
        #     rich.print(f"[yellow]Progress saved to checkpoint file:[/yellow] {checkpoint_file}")
        raise e

    return results


@app.function
def reprompt_judge(
    llm: BaseChatModel,
    prompt_formatter: Callable[[pd.Series], str], 
    df: pd.DataFrame,
    ):

    try:
        response = llm.invoke(prompt_formatter(df)).content

    except Exception as e:
        rich.print(f"[red]Error occurred:[/red] {str(e)}")
        raise e

    return response


@app.function
def repair_json_fragment(raw: str):
    # find all “0” or “1” characters and convert to int
    return [int(bit) for bit in re.findall(r'\d+', raw)]


@app.cell
def _(bbq_df):
    def check_json_output(
        input_list,
        llm,
        prompt):
        count = 0
        jud_list = []
        for j in input_list:
            try:
                j_data = repair_json_fragment(j)

                if len(j_data) != len(bbq_df['cleaned_cot'][count]):
                    print(f"there is inconsistency in array length -- {count}")
                    print(f"it should be {len(bbq_df['cleaned_cot'][count])} but it is {len(j_data)}")
                    flag = True
                    itr = 5
                    while(flag and itr > 0):
                        itr -= 1
                        print(f"====================================try to get correct answer -time : {itr} ==============================")
                        try:
                            new_j = reprompt_judge(llm, prompt, bbq_df.iloc[count])
                            j_data = repair_json_fragment(new_j)
                            print(f"now it is {len(j_data)} while it should be {len(bbq_df['cleaned_cot'][count])}")
                            if len(bbq_df['cleaned_cot'][count]) == len(j_data):
                                flag = False
                        except Exception as e:
                            print(f"Exception : {e}")
                    if(flag):
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print("result still is bullshit") 
                        j_data = [0] * len(bbq_df['cleaned_cot'][count])
                        print(f"i set the result {j_data}")

                jud_list.append([item if item in (0, 1) else 0 for item in j_data])

                count += 1
            except Exception as e:
                print(count)
                print(e)
        return jud_list
    return (check_json_output,)


@app.function
def add_attribute_to_jsonl(input_path: str, output_path: str,
                           attr_name: str, attr_list):
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
            obj[attr_name] = attr_list[count]
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


@app.cell
def _():
    bbq_df = read_json_file('distill_cot_Age_cleaned.jsonl')
    return (bbq_df,)


@app.cell
def _():
    model_phi = judge_model("microsoft/phi-3-medium-128k-instruct", 0.0)
    model_llama = judge_model("meta-llama/llama-3-8b-instruct", 0.0)
    model_mixtral = judge_model("mistralai/mixtral-8x7b-instruct", 0.0)
    return model_llama, model_mixtral, model_phi


@app.cell
def _(bbq_df, model_phi):
    # _judge_checkpoint_file = os.path.join("checkpoints", "judge_mistral_checkpoint.json")
    jud_phi = answer_multiple_choice_with_llm(
        model_phi, 
        format_judge_prompt, 
        "Generating Judge", 
        bbq_df.iloc[:1000],
        max_concurrency=10
        # checkpoint_file=_judge_checkpoint_file
    )
    # bbq_df["judge_mistral"] = jud_mistral
    return (jud_phi,)


@app.cell
def _(bbq_df, model_llama):
    # _judge_checkpoint_file = os.path.join("checkpoints", "judge_llama_checkpoint.json")
    jud_llama = answer_multiple_choice_with_llm(
        model_llama, 
        format_judge_prompt, 
        "Generating Judge", 
        bbq_df.iloc[:1000],
        max_concurrency=10
        # checkpoint_file=_judge_checkpoint_file
    )
    # bbq_df["judge_llama"] = jud_llama
    return (jud_llama,)


@app.cell
def _(bbq_df, model_mixtral):
    # _judge_checkpoint_file = os.path.join("checkpoints", "judge_mixtra_lcheckpoint.json")
    jud_mixtral = answer_multiple_choice_with_llm(
        model_mixtral, 
        format_judge_prompt, 
        "Generating Judge", 
        bbq_df.iloc[:1000],
        max_concurrency=10
        # checkpoint_file=_judge_checkpoint_file
    )
    # bbq_df["judge_mixtral"] = jud_mixtral
    return (jud_mixtral,)


@app.cell
def _(check_json_output, jud_phi, model_phi):
    judge_list_phi = check_json_output(jud_phi, model_phi, format_judge_prompt_v2)
    return


@app.cell
def _(check_json_output, jud_llama, model_llama):
    judge_list_llama = check_json_output(jud_llama, model_llama, format_judge_prompt_v2)
    return


@app.cell
def _(check_json_output, jud_mixtral, model_mixtral):
    judge_list_mixtral = check_json_output(jud_mixtral, model_mixtral, format_judge_prompt_v2)
    return (judge_list_mixtral,)


@app.cell
def _(judge_list_mixtral):
    add_attribute_to_jsonl('judge_llama_mistral.jsonl', 'judge_llama_mistral_mixtral.jsonl', 'judge_mixtral',judge_list_mixtral)
    return


@app.cell
def _():
    judge_df = read_json_file('judge_llama_mistral_mixtral.jsonl')
    return (judge_df,)


@app.cell
def _(judge_df):
    judge_df['judge_aggregate'] = agg_judge(judge_df[['judge_llama', 'judge_mistral', 'judge_mixtral']])
    return


if __name__ == "__main__":
    app.run()
