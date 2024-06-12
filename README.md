# Uncertainty estimation in free-form question answering

Evaluate different uncertainty measures in the context of free-form question answering using the TriviaQA dataset, based
on the work by Kuhn et al. ([2023](https://arxiv.org/pdf/2302.09664)). The
measures explored include:

- Number of semantically distinct answers
- Predictive Entropy
- Semantic Entropy

Files:

- `generate_answers.ipynb:` Explore different prompts and sampling techniques to generate answers.
- `save_generations.py:` Save generated answers along with additional information such as token length for different
  temperatures and sampling algorithms. These answers (stored as pickle files) are later used to
  calculate the
  diversity/uncertainty.
- `post_processing_answers.ipynb:` Post-process the generated pickle files for easier analysis.
- `greedy_answers.ipynb:` Generate and evaluate the correctness of the greedy answers for each question.
- `diviersity_answers.ipynb:` Calculate and analyse the diversity of generated answers.
- `infer_answers.ipynb:` Group semantically equivalent answers into bins for each question.
- `evaulate_uncertainty.ipynb:` Calculate the different uncertainty measures and evaluate them using AUROC.
- `utils.py:` Utility functions.
- `config.yaml:` Configuration parameters.