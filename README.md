# Evaluating Automated Approaches to Measure Moral Foundations in Non-English Discourse

This is the repository for the paper "Evaluating Automated Approaches to Measure Moral Foundations in Non-English Discourse". It is currently under review at the ICWSM 2025. A Prepint of the paper can be found [here on Archive](https://xxx).

## General Takeaways
1. LLMs is the best automated measurment for measuring moral foundations in non-English discourse, compared to translations, lexicon-based approaches, machine learning, and multilingual language models. 
2. Data augmentation can significantly improve the performance of LLMs in measuring moral foundations in non-English discourse via transfer learning. 
3. If researchers have adequate local-language labelled data, fine-tuning a multlingual language model with both English and local-language data can be a good alternative as well for this task. 

## Data
- Chinese datasets
    - MFV - Moral Foundation Vignettes
        - __Source__: Clifford S, Iyengar V, Cabeza R, Sinnott-Armstrong W. Moral foundations vignettes: a standardized stimulus database of scenarios based on moral foundations theory. Behav Res Methods. 2015 Dec;47(4):1178-1198. doi: 10.3758/s13428-014-0551-2. PMID: 25582811; PMCID: PMC4780680.
        - __Access__: MFVs can be openly accessed [in Table 6 ](https://pmc.ncbi.nlm.nih.gov/articles/PMC4780680/#_ci93_)
    - CS - Crowdsourced Moral Scenarios 
        - __Source__: Cheng, C. Y., & Zhang, W. (2023). C-MFD 2.0: Developing a Chinese Moral Foundation Dictionary. Computational Communication Research, 5(2). https://doi.org/10.5117/CCR2023.2.10.CHEN
        - __Access__: the reverse-labelled dataset can be retrieved [here](https://docs.google.com/spreadsheets/d/1z-b2wPezZjbCwqVdala3Knks_-zeE6-HJdXHMVEtfA4/edit?usp=sharing).
    - CV - Chinese CoreValue Dataset
        - __Source__: Pengyuan Liu, Sanle Zhang, Dong Yu, and Lin Bo. 2022. CoreValue: Chinese Core Value-Behavior Frame and Knowledge Base for Value Computing. In Proceedings of the 21st Chinese National Conference on Computational Linguistics, pages 417–430, Nanchang, China. Chinese Information Processing Society of China.
        - __Access__: please email the original authors for access to the original dataset. OR request access to the dataset via the [link provided in the paper](https://docs.google.com/spreadsheets/d/1Zg0mKH5rK9RpVSf61P6nI6vSdxLsp5HW/edit?usp=sharing&ouid=114849464238842402590&rtpof=true&sd=true).

- English datasets
    - Reddit - can be downloaded from [Huggingface dataset](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC)
    - Twitter - TweetID are shared on [OSF](https://osf.io/k5n7y/), but for the full tweet content, please [contact the authors](https://journals.sagepub.com/doi/10.1177/1948550619876629) given the current Twitter API restrictions. 
    - English News - can be downloaded from [OSF](https://osf.io/52qfe)

## Code

### Machine Translation
see code in `e-mt.ipynb` 

### Local Language Dictionary 
- C-MFD2, see code in `e_tool_lexicon.ipynb`. 
- C-MFD2 with sentiment dimension info is saved in `customized.csv`, so it can be used with `FrameAxis` pacakge.
- to make the CMFD2.0 dictionary compatible with `FrameAxis`, revise code in the `FrameAxis` pacakge accordingly (for example, `frameAxis_cmfd2.py` and `frameaxis_main_cmfd2.py`). 




