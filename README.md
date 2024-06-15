# Document classification with HAN, TextGCN and BERT
Document classification with **HAN, TextGCN, BERT** re-implementing project in PyTorch. 

## Datasets
Our dataset processing pipeline expects data corpus being stored in `.csv` format with two fields: `text` and `labels`.

Three datasets are used for our experiments:
- YELP-100k: First 100k samples of YELP Review: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset
- MR: https://huggingface.co/datasets/mattymchen/mr
- R52: https://www.kaggle.com/datasets/weipengfei/ohr8r52

## HAN - Hierarchical attention networks
Our PyTorch implementation for **Hierarchical Attention Networks for Document Classification (NAACL 2016)** can be found at ``han-exps.ipynb``.

<p align="center" height="50%">
    <img scale="45%" height="50%" src="https://github.com/vhminh2210/Document-Classification/blob/main/figs/HAN.png"> 
</p>

Parameters to be modified:
```python
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# Embedding type. 'glove' or 'negative_sampling'
MODE = 'glove'

# Task configs
VOCAB_SIZE = len(vocab_ns)
EMBEDDING_DIM = 200 # For Negative sampling
GRU_DIM = 50
NUM_CLASSES = 5

# Load GloVe embeddings
EMBEDDING_DIM = 300

# Training configs
LR = 1e-3
MOMENTUM = 0.9
EPOCHS = 3
ITER = EPOCHS * len(trainloader)
OPTIMIZER = torch.optim.AdamW(model.parameters(), lr= LR)
LOSS_FN = nn.CrossEntropyLoss()
```

Following original paper suggestions, we also re-implement Negative Sampling introduced in **Distributed Representations of Words and Phrases and their Compositionality (NIPS 2013)**. The pretrained model is used for `MODE = negative_sampling` within `han-exps.ipynb`. However, we recommend using GloVe for HAN. Redudant cells regarding Negative sampling in our Jupyter notebook can be commented out. 

Our Negative sampling re-implementation can be found at `negsampling-exps.ipynb`.
 
Kaggle notebook version can be found at: 
- **HAN**: https://www.kaggle.com/code/vhminh2210/han-exps

## TextGCN - Graph Convolutional Networks for Text Classification
Our PyTorch implementation for **Graph Convolutional Networks for Text Classification (AAAI 2019)** can be found at ``textgcn-exps.ipynb`` which supports basic text graph construction over `MR` and `R52` datasets.

For more complicated tasks and datasets, we ultilized the more stable implementation provided by **Understanding Graph Convolutional Networks for Text Classification (AAAI 2022 on DLG)**. A modified copy of their experiments can be found at `textgcn-ptest.ipynb`. We use this notebook for TextGCN-related experiments.

The original source code can be found at: https://github.com/usydnlp/TextGCN_analysis/blob/main/final.ipynb

<p align="center" width="50%">
    <img scale="45%" width="70%" src="https://github.com/vhminh2210/Document-Classification/blob/main/figs/TextGCN.png"> 
</p>

Parameters to be modified for `textgcn-ptest.ipynb`:
```python
EDGE = 2 # 0:d2w 1:d2w+w2w 2:d2w+w2w+d2d
NODE = 0 # 0:one-hot #1:BERT 
NUM_LAYERS = 2 

HIDDEN_DIM = 200
DROP_OUT = 0.5
LR = 0.02
WEIGHT_DECAY = 0
EARLY_STOPPING = 10
NUM_EPOCHS = 200
```

For `textgcn-exps.ipynb`, `EDGE = 1, NODE = 0, NUM_LAYERS = 2, EARLY_STOPPING = 10` are fixed parameters. Other parameters can be modified.

## BERT - Bidirectional Encoder Representations from Transformers
We investigate the capabilities of BERT for document classification following the configuration introduced by **DocBERT: BERT for Document Classification**.

We used pretrained `PYTORCH-TRANSFORMERS` (https://pytorch.org/hub/huggingface_pytorch-transformers/) models. 

Due to technical constraints, small modifications on training hyperparameters. Our experiments is performed using `bert-base-uncased` model. The source code for BERT experiments is provided at: `bert-exps.ipynb`.

<p align="center" width="50%">
    <img scale="45%" width="70%" src="https://github.com/vhminh2210/Document-Classification/blob/main/figs/BERT.png"> 
</p>

Parameters to be modified:
```python
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_CLASSES = 2
MSL = 256 # Maximum sequence length

# Tokenizer
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
# Pretrained model
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-uncased', num_labels= NUM_CLASSES)
model.to(DEVICE)
```

## References
[1]. Adhikari, Ashutosh, et al. **"Docbert: Bert for document classification."** arXiv preprint arXiv:1904.08398 (2019).

[2]. Devlin, Jacob, et al. **"Bert: Pre-training of deep bidirectional transformers for language understanding."** arXiv preprint arXiv:1810.04805 (2018).

[3]. Han, Soyeon Caren, et al. **"Understanding graph convolutional networks for text classification."** arXiv preprint arXiv:2203.16060 (2022).

[4]. Mikolov, Tomas, et al. **"Distributed representations of words and phrases and their compositionality."** Advances in neural information processing systems 26 (2013).

[5]. Pennington, Jeffrey, Richard Socher, and Christopher D. Manning. **"Glove: Global vectors for word representation."** Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.

[6]. Yang, Zichao, et al. **"Hierarchical attention networks for document classification."** Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. 2016.

[7]. Yao, Liang, Chengsheng Mao, and Yuan Luo. **"Graph convolutional networks for text classification."** Proceedings of the AAAI conference on artificial intelligence. Vol. 33. No. 01. 2019.
