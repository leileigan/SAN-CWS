Investigate self-attention network for Chinese Word Segmentation
====

Investigate self-attention network for Chinese Word Segmentation. 

Models and results can be found at our paper 
[Investigate self-attention network for Chinese Word Segmentation](https://arxiv.org/abs/1907.11512). 


Requirement:
======
	Python: 3.6.2  
	PyTorch: 1.0.1 


Input format:
======
CoNLL format (prefer BMES tag scheme), with each character its label for one line. Sentences are splited with a null line.

	中 B-SEG
	国 E-SEG
	最 B-SEG
	大 E-SEG
	氨 B-SEG
	纶 M-SEG
	丝 E-SEG
	生 B-SEG
	产 E-SEG
	基 B-SEG
	地 E-SEG
	在 S-SEG
	连 B-SEG
	云 M-SEG
	港 E-SEG
	建 B-SEG
	成 E-SEG

	新 B-SEG
	华 M-SEG
	社 E-SEG
	北 B-SEG
	京 E-SEG
	十 B-SEG
	二 M-SEG
	月 E-SEG
	二 B-SEG
	十 M-SEG
	六 M-SEG
	日 E-SEG
	电 S-SEG

Pretrained Embeddings:
====

* Character embeddings: [gigaword_chn.all.a2b.uni.ite50.vec](https://pan.baidu.com/s/1pLO6T9D)
* Character bigram embeddings: [gigaword_chn.all.a2b.bi.ite50.vec](https://pan.baidu.com/s/1pLO6T9D), same folder with character embeddings.

How to run the code?
====
1. Download the character embeddings, character bigram embeddings and set their directories in [`main.py`](https://github.com/jiesutd/SubwordEncoding-CWS/blob/41d6671ecac8209901f77001bf68d33b8b128e01/main.py#L395).
2. Modify the `run_seg.sh` by adding your train/dev/test file directory.
3. `sh run_seg.sh`



Cite: 
========
Cite our paper as:

    @article{gan2019investigating,
      title={Investigating Self-Attention Network for Chinese Word Segmentation},
      author={Gan, Leilei and Zhang, Yue},
      journal={arXiv preprint arXiv:1907.11512},
      year={2019}
    }
