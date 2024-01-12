# BiEL

This repo provides the code for our paper submitted to IJCAI-2024: "MESS: Coarse-grained Modular Two-way Dialogue Entity Linking Framework"

## Setup

```
conda create --name biel python=3.8
conda activate biel
pip install -r requirements.txt
conda install -c pytorch faiss-gpu cudatoolkit=11.0
```

## Datasets

These dataset (except BLINK data) are a pre-processed version of [Phong Le and Ivan Titov (2018)](https://arxiv.org/pdf/1804.10637.pdf) data availabe [here](https://github.com/lephong/mulrel-nel). BLINK data taken from [here](https://github.com/facebookresearch/KILT).

### Entity Disambiguation (train / dev)
- [BLINK train](http://dl.fbaipublicfiles.com/KILT/blink-train-kilt.jsonl) (9,000,000 lines, 11GiB)
- [BLINK dev](http://dl.fbaipublicfiles.com/KILT/blink-dev-kilt.jsonl) (10,000 lines, 13MiB)
- [AIDA-YAGO2 train](http://dl.fbaipublicfiles.com/GENRE/aida-train-kilt.jsonl) (18,448 lines, 56MiB)
- [AIDA-YAGO2 dev](http://dl.fbaipublicfiles.com/GENRE/aida-dev-kilt.jsonl) (4,791 lines, 15MiB)

### Entity Disambiguation (test)
- [ACE2004](http://dl.fbaipublicfiles.com/GENRE/ace2004-test-kilt.jsonl) (257 lines, 850KiB)
- [AQUAINT](http://dl.fbaipublicfiles.com/GENRE/aquaint-test-kilt.jsonl) (727 lines, 2.0MiB)
- [AIDA-YAGO2](http://dl.fbaipublicfiles.com/GENRE/aida-test-kilt.jsonl) (4,485 lines, 14MiB)
- [MSNBC](http://dl.fbaipublicfiles.com/GENRE/msnbc-test-kilt.jsonl) (656 lines, 1.9MiB)
- [WNED-CWEB](http://dl.fbaipublicfiles.com/GENRE/clueweb-test-kilt.jsonl) (11,154 lines, 38MiB)
- [WNED-WIKI](http://dl.fbaipublicfiles.com/GENRE/wiki-test-kilt.jsonl) (6,821 lines, 19MiB)

### Entity Linking (train)
- [WIKI-ABSTRACTS](http://dl.fbaipublicfiles.com/GENRE/train_data_e2eEL.tar.gz) (6,221,563 lines, 5.1GiB)