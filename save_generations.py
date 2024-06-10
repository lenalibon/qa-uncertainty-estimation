from datasets import load_dataset
import torch
from transformers import AutoTokenizer, OPTForCausalLM
import argparse
import os
import pickle
import numpy as np
from utils import calculate_probability_sequence
from tqdm import tqdm
import yaml

parser = argparse.ArgumentParser("Save generations")
parser.add_argument("--specific_group", type=int, default=-1,
                    help="Only create pkl file for specific group (0 - 4). If not in range, for all.")
parser.add_argument("--sampling_method", type=str, default="multinomial_sampling",
                    choices=["multinomial_sampling", "multinomial_beam_sampling"], help="Sampling method to use.")
args = parser.parse_args()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_dir = config["model_dir"]
checkpoint = config["checkpoint"]
max_new_tokens = config["max_output_length"]
temperatures = config["temperatures"]
n_beams = config["n_beams"]
n_samples_per_question = config["n_generations_per_answer"]
specific_group = args.specific_group
sampling_method = args.sampling_method
save_path = config["path_to_saved_generations"]


def create_index_file():
    """
    Creates index file
    :return: None
    """
    if not os.path.exists(os.path.join(save_path, "group_indices.txt")):
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "group_indices.txt"), "w") as f:
            for _ in range(5):
                sampled_indices = np.random.choice(len(data_trivia_val), 1000, replace=False)
                f.write(",".join([str(i) for i in sampled_indices]) + "\n")
    else:
        print("Sampled indices file already exists. Skipping...")


def load_results(filepath):
    """
    Loads pickle file for group if it already exists, otherwise returns new dict
    :param filepath: Filepath to saved generated answers
    :return: Dictionary (maybe already containing generated answers)
    """
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return dict()


# Load data
data_trivia = load_dataset("trivia_qa", "rc.nocontext")
data_trivia = data_trivia.remove_columns(["question_source", "entity_pages", "search_results"])
data_trivia_train = data_trivia["train"]
data_trivia_val = data_trivia["validation"]

# Load and setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{checkpoint}", cache_dir=model_dir)
model = OPTForCausalLM.from_pretrained(f"facebook/{checkpoint}", cache_dir=model_dir)
model = model.to(device)

# Prompt and setup for generation
selected_training_data = data_trivia_train.select(range(0, 10))
ten_shot_prompt = ""
for data in selected_training_data:
    ten_shot_prompt += "QUESTION:" + data["question"] + "ANSWER:" + data["answer"]["value"] + "\n"
stop_tokens = ["Q:", "Question:", "QUESTION:", "questions:", " Q:", " Question:", " QUESTION:", " questions:",
               "A:", "Answer:", "ANSWER:", "answers:", " A:", " Answer:", " ANSWER:", " answers:", "Answers:",
               " Answers:",
               "Topic:", " Topic:", "TOPIC:", " TOPIC:", ".", " .", "...", " ...", "?", " ?", ":", " :", "!", " !"]
stop_tokens = [[tokenizer(stop_token)["input_ids"][1]] for stop_token in stop_tokens]
eos_token = tokenizer("\n")["input_ids"][1]
tokenizer.pad_token_id = eos_token
tokenizer.eos_token_id = eos_token

# Read indices
create_index_file()
with open(os.path.join(save_path, "group_indices.txt"), "r") as f:
    indices_groups = [[int(i) for i in line.strip().split(",")] for line in f]

for group in [specific_group] if 0 <= specific_group <= 4 else range(5):
    print(f"Processing group {group}")
    indices_group = indices_groups[group]
    os.makedirs(os.path.join(save_path, sampling_method), exist_ok=True)
    file_path = os.path.join(save_path, sampling_method, f"group{group}.pkl")
    result = load_results(file_path)

    for i in tqdm(indices_group):
        if i in result:
            print(f"Index {i} is already processed. Skipping...")
            continue
        question = ten_shot_prompt + "QUESTION:" + data_trivia_val[i]["question"] + "ANSWER:"
        answer = data_trivia_val[i]["answer"]["value"]
        inputs = tokenizer(question, padding=False, truncation=False, return_tensors="pt").to(device)
        length_input = inputs["input_ids"].shape[1]

        result_entry = {
            "question": data_trivia_val[i]["question"],
            "true_answer": answer
        }

        if sampling_method == "multinomial_sampling":
            for temp in temperatures:
                temp_key = f"temperature_{str(temp)}"
                result_entry[temp_key] = {"answers": [], "probabilities": [], "length_sequences": []}

                # Sample sequences
                output_generate = model.generate(inputs.input_ids,
                                                 max_new_tokens=max_new_tokens,
                                                 eos_token_id=eos_token,
                                                 bad_words_ids=stop_tokens,
                                                 return_dict_in_generate=True,
                                                 output_scores=True,
                                                 do_sample=True,
                                                 num_return_sequences=n_samples_per_question,
                                                 temperature=temp,
                                                 top_p=0.9)

                for n_sequence in range(n_samples_per_question):
                    output = tokenizer.batch_decode(output_generate.sequences[n_sequence][length_input:],
                                                    skip_special_tokens=True)
                    output_string = "".join(output)
                    result_entry[temp_key]["answers"].append(output_string)

                    # Calculating probability of sequence
                    prob_output, n_output_tokens = calculate_probability_sequence(model, tokenizer, output_generate,
                                                                                  length_input,
                                                                                  idx=n_sequence,
                                                                                  print_scores=False)

                    result_entry[temp_key]["probabilities"].append(prob_output)
                    result_entry[temp_key]["length_sequences"].append(n_output_tokens)

        elif sampling_method == "multinomial_beam_sampling":
            for beam in n_beams:
                beam_key = f"beam_{str(beam)}"
                result_entry[beam_key] = {"answers": [], "probabilities": [], "length_sequences": []}

                # Sample sequence
                output_generate = model.generate(inputs.input_ids,
                                                 max_new_tokens=max_new_tokens,
                                                 eos_token_id=eos_token,
                                                 bad_words_ids=stop_tokens,
                                                 no_repeat_ngram_size=3,
                                                 return_dict_in_generate=True,
                                                 output_scores=True,
                                                 do_sample=True,
                                                 num_beams=beam,
                                                 num_return_sequences=n_samples_per_question)

                for n_sequence in range(n_samples_per_question):
                    output = tokenizer.batch_decode(output_generate.sequences[n_sequence][length_input:],
                                                    skip_special_tokens=True)
                    output_string = "".join(output)
                    result_entry[beam_key]["answers"].append(output_string)

                    # Calculating probability of sequence
                    prob_output, n_output_tokens = calculate_probability_sequence(model, tokenizer, output_generate,
                                                                                  length_input,
                                                                                  beam_sampling=True,
                                                                                  idx=n_sequence,
                                                                                  print_scores=False)
                    result_entry[beam_key]["probabilities"].append(prob_output)
                    result_entry[beam_key]["length_sequences"].append(n_output_tokens)

        # Save result
        result[i] = result_entry
        with open(file_path, "wb") as f:
            pickle.dump(result, f)
