# Can personality traits and demographics predict drug use? Using supervised learning to classify drug consuption behavior.

*python | logistic regression | random forest | SVM classification*

# Intro

One of the first things that fascinated me about machine learning is the ability to predict the behavior of individuals. There are many uses for this capability, from predicting the liklihood of an individual to click on an add, to recommending follow-up mental health treatment based on risk of self-harm. 

## Dataset

Human behavior is delightfully complex. This complexity is reflected in this [drug consumption dataset](http://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29), which contains demographic info as well as scores from personality inventories. This dataset is interesting to me because constructs from personality inventories can often seem abstract, so it's informative to explore how they relate to a concrete behavior. This dataset includes scores from the NEO personality inventory (neuroticism, extraversion, openness to experience, agreeableness, and conscientiousness), the BIS-11 (impulsivity), and ImpSS (sensation seeking). Last, it includes info on whether each individual had used a number of drugs. 

# Project Goals

In this project, I explored the ability of three supervised learning algorithms to predict whether an individual has used a specific drug. I explored and visualized parameter selection for each model, as well as how metrics can help us select the best model for a problem. 

# Setup

- My anaconda venv can be created by running  `conda env create -f environment.yml` in the terminal
- Or, pip install packages with `pip install -r requirements.txt`

# Usage

Clone this repo to view project scripts and Jupyter nb are in the `drug_consumption` directory

- `drug_consumption.ipynb` contains all the code for the project and comments

- Project scripts
    - `settings.py` sets the root directory of the project and makes the project a module
    - `import_data.py` saves and loads data 
    - `clean_data.py` rename some categories and add some features
    - `prepare_data.py` create dataset with one drug target, split into train and test sets
    - `models/wrapper.py` a function to search for the best parameters for a classifier, report a number of metrics for the classifier, refits on a metric of choice, and produces a heatmap for cross validation results
    - `models/rf.py` fits a processing pipeline and Random Forest classifier using `wrapper.py`
    - `models/svc.py` fits a processing pipeline and SVM classifier using `wrapper.py`
    - `models/logreg.py` fits a processing pipeline and Logistic regression classifier using `wrapper.py`
    - `compare_models.py` compares metrics between classifiers, selects a model, visualize ROC curves


# Findings


