import numpy as np
import torch


def calculate_probability_sequence(model, tokenizer, output_generate, length_input, idx=0, beam_sampling=False,
                                   print_scores=False):
    """
    Calculates the probability of a generated sequence, excludes special tokens, EOS token
    :param model: Model used for sampling
    :param tokenizer: Tokenizer used for sampling
    :param output_generate: output of function generate(...)
    :param length_input: length of input_ids to function generate(...)
    :param idx: if multiple sequences got generated, index of the sequence for that probability should be calculated (0 if only one sequence)
    :param beam_sampling: Set true if you want to calculate the probability of a sequence created with beam sampling
    :param print_scores: Set True if table with token, decoded token, score and probability should be printed out
    :return: Probability of generated sequence, number of output tokens
    """

    if not beam_sampling:
        transition_scores = model.compute_transition_scores(output_generate.sequences, output_generate.scores,
                                                            normalize_logits=True)

    else:
        transition_scores = model.compute_transition_scores(output_generate.sequences, output_generate.scores,
                                                            output_generate.beam_indices, normalize_logits=True)

    generated_tokens = output_generate.sequences[:, length_input:]
    prob_output = 1
    n_output_tokens = 0

    for tok, score in zip(generated_tokens[idx], transition_scores[idx]):
        decoded_token = tokenizer.decode(tok)
        if print_scores:
            print(
                f"{tok:5d} | {repr(decoded_token):12s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")

        # Don't include special tokens in generation
        if "<" not in decoded_token or ">" not in decoded_token:
            prob_output *= np.exp(score.cpu().numpy())
            n_output_tokens += 1

    return prob_output, n_output_tokens
