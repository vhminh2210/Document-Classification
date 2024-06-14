# Document classification with HAN, TextGCN and BERT
Document classification with **HAN, TextGCN, BERT** re-implementing project in PyTorch. 

## HAN - Hierarchical attention networks
Our PyTorch implementation for **Hierarchical Attention Networks for Document Classification (NAACL 2016)** can be found at ``han-exps.ipynb``.

<p align="center" width="100%">
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
