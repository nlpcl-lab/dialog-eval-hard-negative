import os
import pickle
import random
from typing import List

import numpy as np
from tqdm import tqdm
import torch
from nltk.corpus import stopwords
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

from utils import set_random_seed


stopwords = stopwords.words("english") + [".", ",", "!", "?", "'", '"']

SCORE_FNAME_TEMPLATE = "./data/scored/{}_sorted_by_condition_mlm_dd.pck"
FINAL_NEGATIVE_FNAME = (
    "./data/negative_dd/dialogues_{}_negative_mask-fill-coherence{}_1.txt"
)

softmax = torch.nn.Softmax(dim=0)


def calculate_score(
    context: List[str],
    response: str,
    bert: BertForMaskedLM,
    tokenizer: BertTokenizer,
    device,
):
    print("=== New example ===")
    print("Original Context: {}".format(context))
    print("Original Response: {}".format(response))
    dialog = context + " " + response
    assert isinstance(dialog, str) and isinstance(response, str)

    encoded_dialog = tokenizer.encode_plus(dialog, return_tensors="pt")
    encoded_dialog = {k: v.to(device) for k, v in encoded_dialog.items()}

    encoded_response = tokenizer.encode_plus(response, return_tensors="pt")
    encoded_response = {k: v.to(device) for k, v in encoded_response.items()}

    response_begin_index_in_dialog = (
        len(encoded_dialog["input_ids"][0]) - len(encoded_response["input_ids"][0]) + 1
    )
    word_list = []
    dialog_score_list, response_score_list, diff_score_list = [], [], []
    for response_token_index in range(len(encoded_response["input_ids"][0]) - 2):
        response_index_in_dialog = response_token_index + response_begin_index_in_dialog
        response_index_in_response = response_token_index + 1

        # Find the original token and check the integrity
        original_token_in_dialog = (
            encoded_dialog["input_ids"][0][response_index_in_dialog].clone().detach()
        )
        original_token_in_response = (
            encoded_response["input_ids"][0][response_index_in_response]
            .clone()
            .detach()
        )
        word_list.append(tokenizer.convert_ids_to_tokens([original_token_in_dialog])[0])
        assert original_token_in_dialog == original_token_in_response

        # Mask the current token in both dialog and response sentence
        encoded_dialog["input_ids"][0][
            response_index_in_dialog
        ] = tokenizer.mask_token_id
        encoded_response["input_ids"][0][
            response_index_in_response
        ] = tokenizer.mask_token_id

        with torch.no_grad():
            dialog_output = bert(**encoded_dialog)[0]
            response_output = bert(**encoded_response)[0]

        score_in_dialog = float(
            softmax(dialog_output[0][response_index_in_dialog])[
                original_token_in_dialog
            ]
            .cpu()
            .detach()
            .numpy()
        )
        score_in_response = float(
            softmax(response_output[0][response_index_in_response])[
                original_token_in_response
            ]
            .cpu()
            .detach()
            .numpy()
        )

        diff_score_list.append(np.log(score_in_dialog) - np.log(score_in_response))
        dialog_score_list.append(np.log(score_in_dialog))
        response_score_list.append(np.log(score_in_response))

    assert (
        len(word_list)
        == len(diff_score_list)
        == len(encoded_response["input_ids"][0]) - 2
    )

    return (
        word_list,
        diff_score_list,
        [dialog_score_list, response_score_list],
    )


def mask_and_fill_by_threshold(
    text: str,
    bert: BertForMaskedLM,
    tokenizer: BertTokenizer,
    threshold: float,
    device,
    mlm_score,
):
    print("=== New example ===")
    print("Original Text: {}".format(text))
    context, response, word_list, score_list, sorted_index = mlm_score
    if score_list is None:
        raise ValueError
    if len(text.split()) < 3:
        return None
    assert len(sorted_index[0]) == len(sorted_index[1]) == len(score_list)
    response = response.strip()
    text = text.strip()
    assert response == text

    if max(score_list) < threshold or len(score_list) < 3:
        return None

    encoded = tokenizer.encode_plus(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    assert len(encoded["input_ids"][0]) == len(word_list) + 2

    masked_token_indices = []
    masked_token_original_list = []
    for idx, score in enumerate(score_list):
        if score >= threshold:
            masked_token_original_list.append(
                encoded["input_ids"][0][idx + 1].clone().detach()
            )
            encoded["input_ids"][0][idx + 1] = tokenizer.mask_token_id
            masked_token_indices.append(idx + 1)

    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = bert(**encoded)[0]

    changed_indices = []
    for mask_order, mask_index in enumerate(masked_token_indices):
        while True:
            decoded_index = torch.argmax(output[0][mask_index]).item()
            if decoded_index not in [masked_token_original_list[mask_order]]:
                break
            output[0][mask_index, decoded_index] = -100
        changed_indices.append(decoded_index)

    for idx, mask_position in enumerate(masked_token_indices):
        encoded["input_ids"][0][mask_position] = changed_indices[idx]

    changed_response = tokenizer.decode(
        encoded["input_ids"][0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    print("Changed: {}".format(changed_response))
    return changed_response


def make_context_and_response_file(setname="train"):
    if setname == "valid":
        setname = "validation"
    with open(
        "./data/ijcnlp_dailydialog/{}/dialogues_{}.txt".format(setname, setname), "r"
    ) as f:
        ls = [line.strip().split("__eou__") for line in f.readlines()]
        ls = [[el.lower() for el in line if len(el.strip()) != 0] for line in ls]

    new_dataset = []

    for idx, line in enumerate(ls):
        context = line[0]
        response = line[1]
        new_dataset.append([context, response, random.sample(ls, 1)[0][-1]])
    return new_dataset


def make_score_file():
    for setname in ["valid", "train"]:
        final_fname = SCORE_FNAME_TEMPLATE.format(setname)

        device = torch.device("cuda")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        bert.eval()
        ls = make_context_and_response_file(setname)
        assert all([len(el) == 3 for el in ls])

        output = []
        for line_index, line in enumerate(tqdm(ls)):
            context, response, _ = [el.strip() for el in line]

            try:
                words, scores, additional_scores = calculate_score(
                    context,
                    response,
                    bert,
                    tokenizer,
                    device,
                )
            except Exception as err:
                print("\n" * 30)
                print(err)
                words, scores, additional_scores = None, None, None

            output.append(
                [
                    context.strip(),
                    response.strip(),
                    words,
                    scores,
                    additional_scores,
                ]
            )
        print("{}/{} is filled".format(len(output), len(ls)))
        os.makedirs(os.path.dirname(final_fname),exist_ok=True)
        with open(final_fname, "wb") as f:
            pickle.dump(output, f)


def make_negative(threshold: float = 0.5):
    for setname in ["valid", "train"]:
        threshold_fname = SCORE_FNAME_TEMPLATE.format(setname)
        with open(threshold_fname, "rb") as f:
            score_data = pickle.load(f)
        final_fname = FINAL_NEGATIVE_FNAME.format(setname, threshold)
        print(final_fname)
        #        assert not os.path.exists(final_fname)

        device = torch.device("cuda")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
        bert.to(device)
        bert.eval()
        ls = make_context_and_response_file(setname=setname)
        assert all([len(el) == 3 for el in ls])

        output = []
        assert len(ls) == len(score_data)
        for line_index, line in enumerate(tqdm(ls)):
            score = score_data[line_index]
            print(f"{line_index}/{len(ls)}")
            context, response, _ = line
            response = response.strip()
            generated_response = mask_and_fill_by_threshold(
                response,
                bert,
                tokenizer,
                threshold,
                device,
                score,
            )
            if generated_response is None:
                generated_response = "[NONE]"

            output.append([context.strip(), response.strip(), generated_response])
        os.makedirs(os.path.dirname(final_fname), exist_ok=True)
        print("{}/{} is filled".format(len(output), len(ls)))
        with open(final_fname, "w") as f:
            for line in output:
                f.write("|||".join(line))
                f.write("\n")


if __name__ == "__main__":
    set_random_seed(42)
    make_score_file()
    make_negative()
