# TOKENIZATION PIPELINE

## Steps in the tokenization pipeline

### Normalization
- Removal of needless whitespace, lowercasing, and/or removing accents.
    - Eg, Unicode normalization

```python
from  transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # or use gpt2 for different tokenizer
print(type(tokenizer.backend_tokenizer))

<class 'tokenizers.Tokenizer'>

print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are yùou?"))

# Output:
# hello how are you?
```

### Pre-tokenization
- Split the text into small entities like words.

```python
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are you?") # bert-base-uncased

# Output:
# [('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (15, 18))]

tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are you?") # gpt2

# Output:
# [('Hello', (0, 5)), ('Ġ', (5, 6)), ('how', (6, 9)), ('Ġare', (9, 12)), ('Ġyou', (12, 15))]

tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?") # T5-small

# Output:
# [('_Hello,', (0, 6)), ('_how', (7, 11)), ('_are', (12, 16)), ('_you?', (17, 22))]
```

- GPT2 uses a special token `Ġ` to indicate the beginning of a new word.
- GPT2 does not ignore multiple spaces.

### Bype Pair Encoding (BPE):
**Step 1:**
- Build a vocabulary of tokens from the training corpus.

 ```json
"hug", "pug", "pun", "bun", "hugs"  > 
["b", "u", "n", "g", "h", "p", "u", "n", "s", "g"] 
 ```

While tokenization, prefer the ordering of merges that leads to the longest token.

#### Wordpiece Tokenization

"word" -> w ##o ##r ##d

- Scoring function: 
$$
\text{score} = \frac{\text{freq\_of\_pair}}{\text{freq\_of\_first\_token} \times \text{freq\_of\_second\_token}}
$$

- if even one of the token is <unk>, the entire word is treated as <unk>

#### Unigram Tokenization

- Start with a very large vocabulary of tokens and remove the least frequent tokens iteratively until the desired vocabulary size is reached.
    - common substrings in pre-tokenization step.
    - Apply BPE to the remaining tokens.

- At each step of the training, the Unigram algorithm computes a loss over the corpus given the current vocabulary.


