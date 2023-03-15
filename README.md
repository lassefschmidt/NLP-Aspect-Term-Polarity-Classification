# Natural Language Processing

Aspect Term Polarity Classifier that predicts opinion polarities (postive, neutral, negative) for 12 different aspect categories.

***
by: Ian Moon, Adel Remadi, Lasse Schmidt

within: MS Data Sciences & Business Analytics

at: CentraleSupélec & ESSEC Business School
***

### 0. Tips

- take aspect category (e.g. FOOD#QUALITY) and transform it into natural language (quality of food)
- input in model could be (term, sentence, aspect category in natural language, sentiment)
- amazing paper that does huge review of current methods (especially section 3.4 provides many references to different models we could try): https://arxiv.org/abs/2203.01054
- very similar task on Turkish restaurant data: https://github.com/EzgiArslan/aspect-based-sentiment-analysis
- exactly the same dataset (but 2018): https://remicnrd.github.io/Aspect-based-sentiment-analysis/
- exactly same dataset (2018 as well): https://github.com/fern35/NLP_aspect-based-sentiment-analysis

### 1. Evaluation

Grading scheme:
- Accuracy on test set: 10pts
- Accuracy on dev set: 3pts
- Exec speed (training and inference): 4 pts
- README file: 3 pts

The first 3 scores are computed automatically with a python script.

Penalties will be applied for:
- Deliverable delays (-1 point per day)
- Group of more than 4: -5 pts per extra member, e.g. group of 6 => penalty = -5 x 2 = -10
- Exec error due to the use of a non-allowed python library: -5 pts
- Exec error due the use of a non compatible version of a library: -5 pts
- Exec error due to non compatible deliverable format (zip) or folder structure: -2 pts
- Other exec errors: penalty will depend on how much effort is required to correct the errors and make the program run successfully.

### 2. Contents of this Readme File
1. Names of the students who contributed to the deliverable (max=4)
2. A clear and detailed description of the implemented classifier (type of classification model, input and feature representation, resources etc.)
3. Accuracy on the dev dataset.
