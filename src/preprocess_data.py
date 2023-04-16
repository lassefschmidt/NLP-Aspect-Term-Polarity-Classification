import pandas as pd

def preprocess(df, config):

    # get relevant settings
    enrich_inputs = config["input_enrichment"]

    # get model name
    plm_name = config["plm_name"]

    # get separator to mark beginning of review sentence
    if plm_name == "bert-base-cased":
        START = "[CLS] "
        SEP = " [SEP] "
        END = " [SEP]"
        MARKER = "[unused99]"
    elif plm_name == "roberta-base" or plm_name == "roberta-large":
        START = "<s> "
        SEP = " </s> "
        END = " </s>"
        MARKER = "<tgt>"
    else:
        print("please review preprocessing function to include this new model!")
        raise NotImplementedError

    # add marker token to target_term
    # example: "pizzas" --> "MARKER pizzas"
    df = (df
        .assign(target_term = lambda df_: [MARKER + " " + t for t in df_.target_term])
    )

    # insert marker token in review sentence before specific occurence of target term
    # example: "The pizzas were delicious!" --> "The MARKER pizzas were delicious!"
    df = (df
        .assign(sentence = lambda df_: [s[:int(loc[0])] + t + s[int(loc[1]):] for s, t, loc in zip(df_.sentence, df_.target_term, list(df_.target_location.str.split(":")))])
    )
    
    # encode labels
    df = df.assign(labels = pd.get_dummies(df.y).values.tolist()) # labels are of form negative, neutral, positive

    # generate and enrich inputs
    if enrich_inputs == "question_sentence_target":

        aspect_in_nl = {
            "AMBIENCE#GENERAL": "What is the ambience like at the $T$?",
            "DRINKS#PRICES": "What are the prices like for the drinks named $T$?",
            "DRINKS#QUALITY": "What is the quality like for the drinks named $T$?",
            "DRINKS#STYLE_OPTIONS": "What are the style options like for the drinks named $T$?",
            "FOOD#PRICES": "What are the prices like for the food named $T$?",
            "FOOD#QUALITY": "What is the quality like for the food named $T$?",
            "FOOD#STYLE_OPTIONS": "What are the style options like for the food named $T$?",
            "LOCATION#GENERAL": "What is the location like for the restaurant named $T$?",
            "RESTAURANT#GENERAL": "What is the overall experience like at the restaurant named $T$?",
            "RESTAURANT#MISCELLANEOUS": "What is the $T$ like at the restaurant?",
            "RESTAURANT#PRICES": "What are the prices like at the restaurant named $T$?",
            "SERVICE#GENERAL": "What is the quality of the service at $T$?",
        }

        df = (df
            .assign(aspect_nl = lambda df_: [aspect_in_nl[cat] for cat in df_.aspect])
            .assign(aspect_nl = lambda df_: [cat_nl.replace("$T$", t) for cat_nl, t in zip(df_.aspect_nl, df_.target_term)])
            .assign(inputs = lambda df_: [START + q + SEP + s + SEP + t + END for q, s, t in zip(df_.aspect_nl, df_.sentence, df_.target_term)])
        )

    elif enrich_inputs == "question_sentence":

        aspect_in_nl = {
            "AMBIENCE#GENERAL": "What is the ambience like at the $T$?",
            "DRINKS#PRICES": "What are the prices like for the drinks named $T$?",
            "DRINKS#QUALITY": "What is the quality like for the drinks named $T$?",
            "DRINKS#STYLE_OPTIONS": "What are the style options like for the drinks named $T$?",
            "FOOD#PRICES": "What are the prices like for the food named $T$?",
            "FOOD#QUALITY": "What is the quality like for the food named $T$?",
            "FOOD#STYLE_OPTIONS": "What are the style options like for the food named $T$?",
            "LOCATION#GENERAL": "What is the location like for the restaurant named $T$?",
            "RESTAURANT#GENERAL": "What is the overall experience like at the restaurant named $T$?",
            "RESTAURANT#MISCELLANEOUS": "What is the $T$ like at the restaurant?",
            "RESTAURANT#PRICES": "What are the prices like at the restaurant named $T$?",
            "SERVICE#GENERAL": "What is the quality of the service at $T$?",
        }

        df = (df
            .assign(aspect_nl = lambda df_: [aspect_in_nl[cat] for cat in df_.aspect])
            .assign(aspect_nl = lambda df_: [cat_nl.replace("$T$", t) for cat_nl, t in zip(df_.aspect_nl, df_.target_term)])
            .assign(inputs = lambda df_: [START + q + SEP + s + END for q, s in zip(df_.aspect_nl, df_.sentence)])
        )
    
    return df[["inputs", "labels"]]