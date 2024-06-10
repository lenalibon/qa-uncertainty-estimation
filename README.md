# Uncertainty estimation in free-form question answering

Evaluate different uncertainty measures in the context of free-form question answering using the TriviaQA dataset, based
on the work by Kuhn et al. ([2023](https://arxiv.org/pdf/2302.09664)). The
measures explored include:

- Number of semantically distinct answers
- Predictive Entropy
- Semantic Entropy

Files:

- `generate_answers.ipynb:` Explore different prompts to generate answers and find the most suitable one. Also explore
  different sampling techniques.
- `save_generations.py:` Save generated answers (with additional information such as token length) for different
  temperatures and sampling algorithms. These generated answers (saved as pickle files) are then used to
  calculate the
  diversity/uncertainty of answers.
- `greedy_answers.ipynb:` Generate the greedy answer for each question and evaluate its correctness.
- `diviersity_answers.ipynb:` Calculate and analyse the diversity of generated answers.
- `infer_answers.ipynb:` Create bins of semantically equivalent answers per question.
- `evaulate_uncertainty.ipynb:` Calculate the different uncertainty measures and evaluate them using AUROC.
- `utils.py:` Utility functions.
- `config.yaml:` Configuration parameters