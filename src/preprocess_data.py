import pandas as pd

def preprocess(df, enrich_inputs = "aspect_target_sentence"):

    # encode labels
    class_dict = {"negative": 0, "neutral": 1, "positive": 2}
    df = df.assign(labels = pd.get_dummies(df.y).values.tolist()) # labels are of form negative, neutral, positive

    # generate and enrich inputs
    if enrich_inputs == "aspect_sentence":
        aspect_in_nl = {
            "AMBIENCE#GENERAL": "How is the general ambience?",
            "DRINKS#PRICES": "How are the prices of the drinks?",
            "DRINKS#QUALITY": "How is the quality of the drinks?",
            "DRINKS#STYLE_OPTIONS": "How are the style options of the drinks?",
            "FOOD#PRICES": "How are the prices of the food?",
            "FOOD#QUALITY": "How is the quality of the food?",
            "FOOD#STYLE_OPTIONS": "How are the style options of the food?",
            "LOCATION#GENERAL": "How is the location of the restaurant?",
            "RESTAURANT#GENERAL": "How is the restaurant overall?",
            "RESTAURANT#MISCELLANEOUS": "Any other aspects that might be relevant?",
            "RESTAURANT#PRICES": "How are the prices of the restaurant?",
            "SERVICE#GENERAL": "How is the service called $T$?",
        }

        df = (df
            .assign(aspect_nl = lambda df_: [aspect_in_nl[cat] for cat in df_.aspect])
            .assign(inputs = lambda df_: df_[["aspect_nl", "sentence"]].apply(lambda x: " [SEP] ".join(x), axis = 1))
        )

    if enrich_inputs == "aspect_target_sentence":
        aspect_in_nl = {
            "AMBIENCE#GENERAL": "How is the general ambience of the $T$?",
            "DRINKS#PRICES": "How are the prices of the drinks called $T$?",
            "DRINKS#QUALITY": "How is the quality of the drinks called $T$?",
            "DRINKS#STYLE_OPTIONS": "How are the style options of the drinks called $T$?",
            "FOOD#PRICES": "How are the prices of the food called $T$?",
            "FOOD#QUALITY": "How is the quality of the food called $T$?",
            "FOOD#STYLE_OPTIONS": "How are the style options of the food called $T$?",
            "LOCATION#GENERAL": "How is the location of the restaurant called $T$?",
            "RESTAURANT#GENERAL": "How is the restaurant called $T$ overall?",
            "RESTAURANT#MISCELLANEOUS": "How is the restaurant's $T$?",
            "RESTAURANT#PRICES": "How are the prices of the restaurant called $T$?",
            "SERVICE#GENERAL": "How is the service called $T$?",
        }

        df = (df
            .assign(aspect_nl = lambda df_: [aspect_in_nl[cat] for cat in df_.aspect])
            .assign(aspect_nl = lambda df_: [cat_nl.replace("$T$", t) for cat_nl, t in zip(df_.aspect_nl, df_.target_term)])
            .assign(inputs = lambda df_: df_[["aspect_nl", "sentence"]].apply(lambda x: " [SEP] ".join(x), axis = 1))
        )

    return df[["inputs", "labels"]]