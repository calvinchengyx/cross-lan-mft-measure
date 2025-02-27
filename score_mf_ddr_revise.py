import spacy
from spacy.tokens import Doc
from spacy.lang.zh import Chinese
from tqdm import tqdm
import pandas as pd
import numpy as np
# from utils import make_concepts_from_lexicon
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors  # for loading fasttext vectors

import argparse
import os
import warnings
from typing import Iterable, Dict, List
import gensim

import fasttext.util
import requests #for downloading the lexicon
from io import StringIO # for downloading the lexicon

# to run on gpus
import torch
import pandas as pd

fasttext.util.download_model('zh', if_exists='ignore')
binary_model_path = "cc.zh.300.bin"
embedding = load_facebook_vectors(binary_model_path)
# # no_header=True because the glove vector file doesn't have the
# # (number of words, dimension) header default for a typical word2vec file
# embedding = KeyedVectors.load_word2vec_format(
#     emb_path,
#     binary=True,
#     no_header=True
# )
# nlp = spacy.load("en_core_web_md")
nlp = Chinese()

# Hand-craft a lexicon
# lexicon = {
#     "care": ["kindness", "compassion", "nurture", "empathy", "suffer", "cruel", "hurt", "harm"],
#     "fairness": ["fairness", "equality", "patriot", "fidelity", "cheat", "fraud", "unfair", "injustice"],
#     "loyalty": ["loyal", "team", "patriot", "fidelity", "betray", "treason", "disloyal", "traitor"],
#     "authority": ["authority", "obey", "respect", "tradition", "subversion", "disobey", "disrespect", "chaos"],
#     "sanctity": ["purity", "sanctity", "sacred", "wholesome", "impurity", "depravity", "degradation", "unnatural"]
# }

# load cmfd2.0 lexicon
url = "https://raw.githubusercontent.com/CivicTechLab/CMFD/main/cmfd_civictech.csv"
response = requests.get(url)
csv_content = StringIO(response.text)
df = pd.read_csv(csv_content)
lexicon = {
    "care": df[df['foundation'] == 'care']['chinese'].tolist(),
    "fairness": df[df['foundation'] == 'fair']['chinese'].tolist(),
    "loyalty": df[df['foundation'] == 'loya']['chinese'].tolist(),
    "authority": df[df['foundation'] == 'auth']['chinese'].tolist(),
    "sanctity": df[df['foundation'] == 'sanc']['chinese'].tolist()
}

#### Create concept vectors #####
def make_one_concept(model: KeyedVectors,
                     word_list: Iterable[str],
                     concept_name: str = "",
                     normalize: bool = True) -> np.ndarray:
    """
    Create a concept vector from a list of words.
    :param model: Word embedding model. KeyedVectors object.
    :param word_list: List of words describing the concept.
    :param concept_name: (Optional) Name of the concept.
    :param normalize: Whether to normalize the concept vector by its l2 norm (True) or not (False).
    :return: A d-dimensional concept vector aggregated from the vectors in wordlist.
    """
    dim = model.vector_size
    concept_vector = np.zeros(dim, dtype=float)
    word_list = list(set(word_list))
    count = 0
    for w in word_list:
        if w not in model:
            continue
        concept_vector += model[w]
        count += 1
    if count == 0:
        warnings.warn(f"No word in concept '{concept_name}' found in embedding model.")
    if normalize is True and count > 0:
        concept_vector /= np.linalg.norm(concept_vector)
    return concept_vector


def make_concepts_from_lexicon(model: KeyedVectors,
                               lexicon: Dict[str, List[str]],
                               normalize: bool = True,
                               verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Create a dictionary of concept vectors form a lexicon and embedding model.
    :param model: Word embedding model. KeyedVectors object.
    :param lexicon: Lexicon in {concept: [word1, word2]} format.
    :param normalize: Whether to normalize the concept vector by its l2 norm (True) or not (False).
    :param verbose: Whether to print messages (True) or not (False).
    :return: A d-dimensional concept vector aggregated from the vectors in wordlist.
    """
    concepts = {}
    for concept, words in lexicon.items():
        if verbose:
            print(f"Creating vector for concept {concept}...")
        concepts[concept] = make_one_concept(model=model,
                                             word_list=words,
                                             normalize=normalize,
                                             concept_name=concept)
    return concepts

#### Create concept vectors #####

# Create concept vectors
concepts = make_concepts_from_lexicon(model=embedding,
                                      lexicon=lexicon, 
                                      verbose=False, 
                                      normalize=True)


def tokenize(text):
    doc = nlp(text)
    return [tok.text.strip() for tok in doc]


def score_tokens(document, concepts, embedding):
    def score_one_concept(tokens_vectors, concept):
        sims = embedding.cosine_similarities(concept, tokens_vectors)
        return np.mean(sims)

    if type(document) == str:
        tokens = tokenize(document)
    else:
        tokens = document
    tokens_vectors = []
    for w in tokens:
        if w not in embedding:
            continue
        w = embedding[w]
        tokens_vectors.append(w)

    if len(tokens_vectors) <= 0:
        tokens_vectors.append(np.zeros(embedding["x"].shape))

    scores = {}
    for concept, concept_vector in concepts.items():
        scores[concept] = score_one_concept(concept=concept_vector,
                                            tokens_vectors=tokens_vectors)
    scores = pd.Series(scores)
    return scores


def predict_df(df, text_col, output_path):
    df[[f"{f}_score" for f in concepts.keys()]] = 0
    for i, row in tqdm(df.iterrows(), desc="Scored", dynamic_ncols=True,
                       unit=" examples", leave=False, total=len(df)):
        scores = score_tokens(row[text_col], concepts, embedding)
        for concept, score in scores.items():
            df.loc[i, f"{concept}_score"] = score
    df.to_csv(output_path)


def parse_args():
    parser = argparse.ArgumentParser("Score texts using embedding similarity.")
    parser.add_argument("--data",
                        type=str,
                        help="Path to the data file (CSV).",
                        required=True)
    parser.add_argument("--text_col",
                        type=str,
                        help="Name of the column in the data file that contains the texts.",
                        required=True)
    parser.add_argument("--verbose",
                        type=int,
                        help="Whether to print messages (1) or not (0).",
                        default=1)
    parser.add_argument("--output",
                        type=str,
                        help="Path to the output file (CSV).",
                        required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data
    text_col = args.text_col
    verbose = bool(args.verbose)
    output_path = args.output

    # Check if the data file exists
    assert os.path.exists(data_path), f"Data file does not exist at {data_path}."

    # Check if the output file exists
    if os.path.exists(output_path):
        warnings.warn(f"Output file already exists at {output_path}. It will be overwritten.")

    # Load the texts
    if verbose:
        print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Score the texts
    if verbose:
        print(f"Scoring texts...")

    predict_df(df=df,
               text_col=text_col,
               output_path=output_path
               )