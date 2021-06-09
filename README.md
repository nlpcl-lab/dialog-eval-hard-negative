# Negative Sample Generation with BERT
Code for "Generating Negative Samples by Manipulating Golden Responsesfor Unsupervised Learning of a Response Evaluation Model (NAACL-HLT 2021)"

## Quick Start
```
# Python version is 3.8.5
pip install -r requirements.txt

# Dowonload the DailyDialog corpus
python get_dataset.py

# After download DailyDialog corpus, please manually unzip the corpus in './data/ijcnlp_dailydialog.zip'.

# Negative sample generation with BERT. The output file is './data/negative_dd/'
python mask_and_fill_by_bert.py
```

## Citation
[paper](https://www.aclweb.org/anthology/2021.naacl-main.120/)

```bibtex
@inproceedings{park-etal-2021-generating,
    title = "Generating Negative Samples by Manipulating Golden Responses for Unsupervised Learning of a Response Evaluation Model",
    author = "Park, ChaeHun  and
      Jang, Eugene  and
      Yang, Wonsuk  and
      Park, Jong",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.120",
    pages = "1525--1534",
}```

