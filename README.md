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

## Methods
_NOTE_. All codes here we removed the `path` information for privacy reasons. Please be extra cautious when running the code and change it accordingly.

### Machine Translation Approach
see code in `e_mt.ipynb` 

### Local Language Dictionary Approach
- C-MFD2, see code in `e_tool_lexicon.ipynb`. 
- C-MFD2 with sentiment dimension info is saved in `customized.csv`, so it can be used with `FrameAxis` pacakge.
- to make the CMFD2.0 dictionary compatible with `FrameAxis`, revise code in the `FrameAxis` pacakge accordingly (for example, `frameAxis_cmfd2.py` and `frameaxis_main_cmfd2.py`). 

### Multilingual Language Model Approach
see code in `e_tools_lm.ipynb` for data analysis; `xlm_base.py` and `xlm_model_infer.py` files for model fine-tuning and inference.

### LLMs Llama3.1-8b-instruct 

see code in `e_tool_llms.ipynb` for data analysis; `llama_ft.py` and `llama_inference.py` files for sample model fine-tuning and inference.

Why choose Llama3.1 as the representation LLM for this task? The key reason is to set a conservative baseline for the llms approach. 

As an LLM, `llama3.1` is not the best for cross-lingual tasks, because Llama3.1 does not technically support Chinese complex tasks. As stated in its [model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md), the majority training data is English. officially claimed to support: English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai. But, it performs pretty well on [multilingual leadering board](https://huggingface.co/microsoft/Phi-3.5-mini-instruct), though not as good as `gpt4o`. Llama 3 support Chinese though, however, [over 5% of the Llama 3 pretraining dataset consists of high-quality non-English data that covers over 30 languages. However, we do not expect the same level of performance in these languages as in English.](https://ai.meta.com/blog/meta-llama-3/). So, Chinese may not perform well in Llama3 either. There is only [limited language support](https://github.com/meta-llama/llama/issues/58) on Chinese, only 700 characters in the tokenizer. 

Therefore, our hypothesis is, if `llama3.1` can perform well on this task, then it is a good sign that the llms approach is promising.

We ran all llms relevant code on the remote server, with a single L40S GPU. English and Chinese training data are the same as fine-tuning XLM-T model. 

For llms fine-tuning code, please see a base code in `llama_ft.py`, including prompts and fine-tuning details. Experiments on different models are different regarding the base model (llama3.1-8b, 70b; llama3.2-1b, 3b), prompt (Chinese or English) and training data (Chinese or English). Parameters and other variables remain the same.

`llama_inference.py` is the code for inference, including the code for the evaluation of the model.

## Results
The `.ipynb` files contain most of the data analysis and visualization results for the paper. For detailed data/results for replication purposes, please contact the authors, and I am happy to share full paper data in `.csv` or `.json` files. 