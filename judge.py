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
    from prompts import format_mistral_judge_prompt
    from answer_questions import save_checkpoint, load_checkpoint
    from langchain_deepseek import ChatDeepSeek


    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)


@app.cell
def _():
    model_llama = ChatOpenAI(
        openai_api_key=os.environ["judge_key"],
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="meta-llama/llama-3-8b-instruct",
    )
    return (model_llama,)


@app.cell
def _():
    model_mistral = ChatOpenAI(
        openai_api_key=os.environ["judge_key"],
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="mistralai/mistral-7b-instruct",
    )
    return (model_mistral,)


@app.cell
def _():
    # Read the JSONL file
    with open('reasoning_steps.jsonl', 'r') as file:
        data = [json.loads(line) for line in file]

    bbq_df = pd.DataFrame(data)
    bbq_df
    return (bbq_df,)


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

    try:
         # Process dataframe in chunks with progress bar
         for i in tqdm(range(0, len(df), max_concurrency), desc=desc):
             chunk = df.iloc[i:i+max_concurrency]
             # Create prompts for this chunk
             chunk_prompts = [prompt_formatter(bias_question_data) for _, bias_question_data in chunk.iterrows()]
             save_checkpoint = chunk_prompts
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
        if checkpoint_file:
            rich.print(f"[yellow]Progress saved to checkpoint file:[/yellow] {checkpoint_file}")
        raise e

    return results


@app.cell
def _(bbq_df, format_qwen_judge_prompt, model_llama):
    _judge_checkpoint_file = os.path.join("checkpoints", "judge_checkpoint.json")
    jud_llama = answer_multiple_choice_with_llm(
        model_llama, 
        format_qwen_judge_prompt, 
        "Generating Judge", 
        bbq_df,
        max_concurrency=10,
        checkpoint_file=_judge_checkpoint_file
    )
    bbq_df["judge_llama"] = jud_llama

    return (jud_llama,)


@app.cell
def _(bbq_df, model_mistral):
    _judge_checkpoint_file = os.path.join("checkpoints", "judge_checkpoint.json")
    jud_mistral = answer_multiple_choice_with_llm(
        model_mistral, 
        format_mistral_judge_prompt, 
        "Generating Judge", 
        bbq_df,
        max_concurrency=10,
        checkpoint_file=_judge_checkpoint_file
    )
    bbq_df["judge_mistral"] = jud_mistral
    return (jud_mistral,)


@app.cell
def _(jud_mistral):
    jud_mistral
    return


@app.function
def repair_json_fragment(raw: str):
    s = raw.strip()

    # Remove markdown fences
    s = re.sub(r'^```json\s*|```$', '', s, flags=re.MULTILINE).strip()

    # If wrapped in quotes, unescape and remove them
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
        s = s.encode('utf-8').decode('unicode_escape')
        if '\\"' in s:
            s = s.replace('\\"', '"')

    # Remove any control characters
    s = re.sub(r'[\x00-\x1f\x7f]', '', s)

    # Add missing commas between objects in arrays: }{ => },{
    s = re.sub(r'\}\s*\{', '},{', s)

    # Add missing commas between an array close and an object: ]{ => ],{
    s = re.sub(r'\]\s*\{', '],[', s)

    # Reorder any number of closing braces before array close:
    # e.g. '}}]' -> '}]}', '}}}}]' -> '}]}]}'
    def _reorder(match):
        braces = match.group(1)
        count  = braces.count('}')
        # one } before ], then the remaining } after
        return '}' + ']' + '}'*(count-1)
    s = re.sub(r'((?:\}\s*)+)\]', _reorder, s)

    # Remove any trailing commas before } or ]
    s = re.sub(r',\s*(?=[\}\]])', '', s)

    # Remove any trailing non-JSON content (e.g., stray numbers, text)
    last_brace = max(s.rfind('}'), s.rfind(']'))
    if last_brace != -1:
        s = s[:last_brace+1]

    # Try to close the array and object if not closed
    if s.count('[') > s.count(']'):
        # s += ']'
        s = s[:-1] + ']' + s[-1]
    if s.count('{') > s.count('}'):
        s += '}'

    # Final parse-attempt
    try:
        return json.loads(s)
    except Exception:
        print("Failed to parse JSON. Intermediate string:")
        print(s)
        raise


@app.cell
def _(bbq_df):
    def check_json_output(input_list):
        count = 0
        jud_list = []
        for j in input_list:
            try:
                j_data = repair_json_fragment(j)

                if len(j_data['steps']) != len(bbq_df['reasoning_steps'][count]):
                    print(count)
                    print("there is inconsistency in array length")

                jud_list.append([items['score'] for items in j_data['steps']])
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


@app.cell
def _(check_json_output, jud_llama):
    _judge_list = check_json_output(jud_llama)
    # add_attribute_to_jsonl('judge.jsonl', 'judge.jsonl', 'judge_llama',_judge_list)
    return


@app.cell
def _(check_json_output, jud_mistral):
    _judge_list = check_json_output(jud_mistral)
    add_attribute_to_jsonl('reasoning_steps.jsonl', 'judge.jsonl', 'judge_mistral',_judge_list)
    return


if __name__ == "__main__":
    app.run()
