# 13th September - Friday

### Problem:
Existing word embeddings were applied in a context free manner
- Open a BANK account
- On a river BANK
Both the BANK have embeddings irrespective of the context

### Solution:
Train Contextualized Representation on text corpus

### Pretraining vs Finetuning

| Pretraining | Finetuning |
| ------------ | ----------- |
| Train on large corpus | Train on small corpus |
| All unlabelled data | All labelled data |


## BERT:
Input is a sequence of tokens, but two tokens are special tokens that are added.
- CLS - Classification Token
- SEP - Seperation Token

Input always has 2 sentences and has the format of:
> \[CLS\] Sentence 1 \[SEP\] Sentence 2 \[SEP\]

### Formation of Embeddings:
Each Token is the sum of three embeddings
> Final Embedding = Token Embeddings + Segment Embeddings + Position Embeddings
- Token Embeddings: Defines the token at a given position.
- Segment Embeddings: Defines which sentence the token belongs to. First or Second
- Position Embeddings: Defines the position of the token in the sequence.

### Pretraining Objective (Masked LM):
- Mask out k% of the input owrds, and then predict the masked words
    - The man went to the \[Mask\] to buy a \[Mask\] of milk.
    - Store Gallon
- Too litle masking: Too expensive to train
- Too much masking: Not Enough context
> Vocublary has [CLS], [SEP] and [Mask] and specific embeggings for the same


- k% words are candidates for the Masked Word, but not always they will be masked. Instead:
    - 80% of the time, replace the word with \[MASK\]
    - 10% of the time, replace the word with a random word
    - 10% of the time, keep the word as it is

This helps in understanding the context of the sentence at the Masked Position
- The Masked word could be _Bank_ for instance and the context of that word could be understood from the context of the sentence.
- Error is calculated on the k% masked words and not the rest of the sentence.

# 20th September - Friday

### Dimension Analysis of [CLE]

- If d<sub>model</sub> (Embedding size of each word) is 768 then embedding size E<sub>[CLS]</sub> is also 768. But output of [CLS] is binary output.
- So we use a Weight matrix W (size 768x2) to give the binary output and then perform softmax to get total sum equal to unity.
- We then perform cross entropy loss for learning. Ideally...
$$ 
CLE = - \sum_{}^{} p_{actual} . \log({p_{predicted}})
$$
but in our case...
$$
CLE = -\log(p_{predicted})
$$

Because for binary output, only one of the p<sub>actual</sub> will be 1 and rest all will be zero. We will thus get only that p<sub>predicted</sub> corresponding to that.

### Dimension Analysis for other words (MLM)
- Simillarly for other words, suppose we have |v| as the size of the output, we multiply the embeddings with a weight matrix of size $ 768x|v| $ and then perform softmax to get probability distribution. However the actual probability distribution will have one p<sub>actual</sub> will be 1 and rest all will be 0 (becaused only a fraction of the words are masked)
- Thus cross entropy loss will then be for position in |v| where p<sub>actual</sub> is 1.
$$
CLE = -\log(p_{predicted})
$$


### Model Architecture
- BERT-Base: 12-layer, 768-hidden, 12-head
- BERT-Large: 24-layer, 1024-hidden, 16-head

### Fine Tuning
- Done so far is just calculating embeddings to learn according to context. However, how do we make BERT perform some task? **Fine Tunning**
