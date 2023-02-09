# Checklist for Scipy 2023 submission

## Guideline

### General information
- Deadline: February 22, 2023
- Track: Machine Learning, Data Science, and Ethics in AI

### Submission details
- Abstract: ca. 100 words
- Description
  - Word limit: ca. 500 words
  - Content	
    - Software of interest
    - Tools or techniques for more effective computing
    - how scientific Python was applied to solve a research problem
  - Structure: background/motivation, methods, results, and conclusion
- Links
  - Websites
    - IEEE SOCC 2022 paper: publication for the Part 1
    - Scipy 2019: background information
  - Source code repositories: GitHub
  - Figures
    - Part 1
      - Changes in no. features over time
      - Flowchart of the train/serving pipelines
      - Performance comparison: speed and model performance
    - Part 2
      - Benchmark results for data tuning
      - Fail signature comparison between existing and the data-tuned
  - Evidence of public speaking ability
    - Austin Python meetup talk on YouTube

### Tips
- Audience (a broad range of people)
- Takeaways
- Links to source code, articles, blog posts, or other writing that adds context to the presentation
- Previous talk, tutorial, or other presentation information

## Abstract
- Part 1: data preprocessing pipeline for non-stationary data
- Part 2: data tuning

## Datasets
- Synthetic and non-proprietary
- High dimensionality
- High heterogeneity
- High sparsity
- Type changes over time
- No. features changes over time
- Timestamps
- Object data types including lists
- Arbitrary feature names (represent that it's difficult to understand the meaning of the features)

## Code
### Notebook
- Comparing type inference between pandas and numpy
- Demo of the entire flow
- Data tuning
- Caching and memoization

### Code
- Environment
- Modules
    - Schema inference
    - Resolving mismatched during post-deployment
- Prefect flow