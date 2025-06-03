import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import json
    import argparse
    import json
    import os
    import pickle
    import random
    import string
    from tqdm import tqdm

    return json, mo, pd, tqdm


@app.cell
def _(get_samples, mo):
    # from mbbq import get_samples

    # For just one subset:
    datasets = [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Physical_appearance",
        "SES",
        "Sexual_orientation",
    ]
    languages = ["en", "es", "nl", "tr"]
    controls = [True, False]
    for dataset in datasets:
        for control in controls:
            for language in languages:
                mo.output.append(dataset + " " + language + " " + str(control))
                print(dataset + " " + language + " " + str(control))
                _df = get_samples([dataset], control=control, language=language)
                # mo.output.append(_df)
                # save to jsonl in our_datasets
                _df.to_json(
                    f"datasets/our_datasets/{dataset}{'_control' if control else ''}_{language}.jsonl",
                    orient="records",
                    lines=True,
                )
    # This will process data/Age_control_en.jsonl and add target_loc and unknown_loc

    # df
    return


@app.cell
def _(json, pd, tqdm):
    def get_samples(subsets, control=False, language="en", en_prompts=["test"]):
        """Create DataFrame of control or regular samples, using all 5 prompts.

        Keyword arguments:
        subsets -- list of subsets from which to include samples
        control -- whether to evaluate on samples from control BBQ (default False)
        language -- language of samples (default en)
        """
        df_samples = pd.DataFrame([])
        for subset in subsets:
            print(subset)
            with open(
                f"datasets/orig_datasets/{subset}{'_control' if control else ''}_{language}.jsonl"
            ) as infile:
                dataset = [json.loads(jline) for jline in infile]
            samples = []
            contexts = []
            questions = []
            ans0s = []
            ans1s = []
            ans2s = []
            question_polarities = []
            additional_metadatas = []
            answer_infos = []

            example_id = []
            q_id = []
            context_condition = []
            labels = []
            model_answers = []
            unknown_locs = []
            prompt_id = []
            target_loc = []
            for sample in tqdm(dataset):
                # if language != "en":
                #     prompts = [trans_prompt_dict[language][prompt] for prompt in en_prompts]
                # else:
                #     prompts = en_prompts
                # obtain samples using all 5 prompts
                # samples += [
                #     prompts[i]
                #     .strip()
                #     .format(
                #         sample["context"],
                #         sample["question"],
                #         sample["ans0"],
                #         sample["ans1"],
                #         sample["ans2"],
                #     )
                #     for i in range(len(prompts))
                # ]
                contexts.append(sample["context"])
                questions.append(sample["question"])
                ans0s.append(sample["ans0"])
                ans1s.append(sample["ans1"])
                ans2s.append(sample["ans2"])
                question_polarities.append(sample["question_polarity"])
                additional_metadatas.append(sample["additional_metadata"])
                answer_infos.append(sample["answer_info"])
                # make sure entries in 'stereotyped groups' field match 'answer info'
                # to find location of target answer
                if subset == "Gender_identity" and sample["additional_metadata"][
                    "stereotyped_groups"
                ] == ["F"]:
                    sample["additional_metadata"]["stereotyped_groups"] = [
                        "girl",
                        "woman",
                        "vrouw",
                        "meisje",
                        "kız",
                        "kadın",
                        "chica",
                        "mujer",
                        "maestra",
                    ]
                if subset == "Gender_identity" and sample["additional_metadata"][
                    "stereotyped_groups"
                ] == ["M"]:
                    sample["additional_metadata"]["stereotyped_groups"] = [
                        "boy",
                        "man",
                        "jongen",
                        "erkek çocuk",
                        "adam",
                        "chico",
                        "hombre",
                        "maestro",
                    ]
                if subset == "SES" and sample["additional_metadata"][
                    "stereotyped_groups"
                ] == ["low SES"]:
                    sample["additional_metadata"]["stereotyped_groups"] = ["lowSES"]
                # find location of target and unknown answers
                found_target_loc = False
                for ans in sample["answer_info"].items():
                    # make sure entries in 'answer info' field match
                    # 'stereotyped groups'
                    if ans[1][1] == "F":
                        ans[1][1] = "woman"
                    if ans[1][1] == "M":
                        ans[1][1] = "man"
                    if ans[1][1] == "unknown":
                        for _ in range(1):
                            unknown_locs.append(int(ans[0][-1]))
                    if ans[1][1] in sample["additional_metadata"]["stereotyped_groups"]:
                        # a few BBQ samples include two people from target groups
                        if found_target_loc:
                            print("Begin")
                            print(sample)
                            print(ans[1][1])
                            print(sample["additional_metadata"]["stereotyped_groups"])
                            print("end")
                            target_loc = target_loc[:-1]
                            for _ in range(1):
                                target_loc.append(100)
                        else:
                            found_target_loc = True
                            for _ in range(1):
                                target_loc.append(int(ans[0][-1]))
                # a few BBQ samples include two people from non-target groups
                if not found_target_loc:
                    for _ in range(1):
                        target_loc.append(100)

                # across the 5 prompts, the example_id, q_id, context condition
                # label, and model answers are the same, the prompt_id indicates
                # which prompt is used
                for i in range(1):
                    example_id.append(sample["example_id"])
                    q_id.append(int(sample["question_index"]))
                    context_condition.append(sample["context_condition"])
                    labels.append(sample["label"])
                    model_answers.append(
                        [sample["ans0"], sample["ans1"], sample["ans2"]]
                    )
                    prompt_id.append(i)

            # print("Habib:")
            # print("example ids", len(example_id))
            # print("q ids", len(q_id))
            # print("contexts", len(contexts))
            # print("questions", len(questions))
            # print("ans0s", len(ans0s))
            # print("ans1s", len(ans1s))
            # print("ans2s", len(ans2s))
            # print("question polarities", len(question_polarities))
            # print("additional metadatas", len(additional_metadatas))
            # print("answer infos", len(answer_infos))
            # print("unknown locs", len(unknown_locs))
            # print("target locs", len(target_loc))
            # print("prompt ids", len(prompt_id))

            df_samples = pd.concat(
                [
                    df_samples,
                    pd.DataFrame(
                        {
                            "language": [language] * len(contexts),
                            "example_id": example_id,
                            "question_index": q_id,
                            "question_polarity": question_polarities,
                            # "question": samples,
                            # "model_ans": model_answers,
                            "context_condition": context_condition,
                            "category": [subset] * len(contexts),
                            "answer_info": answer_infos,
                            "additional_metadata": additional_metadatas,
                            "unknown_label": unknown_locs,
                            # "prompt_id": prompt_id,
                            "target_loc": target_loc,
                            "context": contexts,
                            "question": questions,
                            "ans0": ans0s,
                            "ans1": ans1s,
                            "ans2": ans2s,
                            "label": labels,
                            "label_type": ["dummy"] * len(contexts),
                        }
                    ),
                ],
                ignore_index=True,
            )
        return df_samples

    return (get_samples,)


if __name__ == "__main__":
    app.run()
