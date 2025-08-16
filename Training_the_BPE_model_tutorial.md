# Byte Pair Encoding (BPE) Tokenization: Complete Guide

## 📚 Table of Contents
1. [What is BPE?](#what-is-bpe)
2. [Why Use BPE?](#why-use-bpe)
3. [How Does BPE Algorithm Work?](#how-does-bpe-algorithm-work)
4. [Code Implementation](#code-implementation)
5. [Step-by-Step BPE Training](#step-by-step-bpe-training)
6. [Tokenization Process](#tokenization-process)
7. [Results Analysis](#results-analysis)
8. [Practical Application](#practical-application)
9. [Advanced Topics](#advanced-topics)

---

## 🤔 What is BPE?

**Byte Pair Encoding (BPE)** is a **subword tokenization** algorithm used in natural language processing. Originally developed for data compression, it was later adapted for NLP applications.

### Core Principles:
- **Subword tokenization**: Splits words into smaller meaningful units
- **Data-driven**: Learns from corpus statistics
- **Frequency-based**: Merges most frequent character pairs

### Example:
```
Input: "unhappiness"
BPE Output: ["un", "happy", "ness"]
```

---

## 🎯 Why Use BPE?

### 1. **Out-of-Vocabulary (OOV) Problem Solution**
```python
# Traditional word-level tokenization
vocab = ["happy", "sad", "good"]
# "unhappy" is not in vocabulary → OOV problem

# With BPE
# "unhappy" → ["un", "happy"] 
# Both parts are in vocab!
```

### 2. **Vocabulary Size Control**
- Word-level: 50,000+ words
- Character-level: ~100 characters
- **BPE: 10,000-50,000 subwords** (configurable)

### 3. **Morphological Understanding**
```
"playing" → ["play", "ing"]
"unhappy" → ["un", "happy"]  
"cats" → ["cat", "s"]
```

### 4. **Multilingual Support**
- Common morphemes across languages
- Code-switching scenarios
- Effective for low-resource languages

---

## ⚙️ How Does BPE Algorithm Work?

### Step 1: Initial Vocabulary
```python
# Words are split into characters
"low" → "l o w"
"lower" → "l o w e r"
"newest" → "n e w e s t"
```

### Step 2: Pair Counting
```python
# Count most frequent character pairs
("l", "o"): 2 times
("o", "w"): 2 times  
("e", "s"): 1 time
```

### Step 3: Merge Most Frequent Pair
```python
# Most frequent pair: ("l", "o") → "lo"
"low" → "lo w"
"lower" → "lo w e r"
```

### Step 4: Repeat
This process continues until target vocabulary size is reached.

---

## 💻 Code Implementation

### Class Structure Overview

```python
class BPETokenizer:
    def __init__(self, vocab_size=5000, unk_token="<UNK>"):
        self.vocab_size = vocab_size      # Target vocabulary size
        self.unk_token = unk_token        # Token for unknown characters
        self.merges = []                  # Learned merge operations
        self.charset = set()              # Known characters
        self.tokens = set()               # All tokens
        self.merge_map = {}               # For fast tokenization
```

### Main Methods:

1. **`train(corpus)`**: Trains BPE on corpus
2. **`tokenize(word)`**: Tokenizes a single word
3. **`save(filename)`**: Saves trained model
4. **`load(filename)`**: Loads saved model

---

## 🔬 Step-by-Step BPE Training

### 1. Vocabulary Initialization
```python
def initialize_vocabulary(self, corpus):
    vocabulary = defaultdict(int)
    charset = set()

    for word in corpus:
        # Add word boundary marker
        word_with_marker = '_' + word  # "_hello"
        characters = list(word_with_marker)  # ['_', 'h', 'e', 'l', 'l', 'o']
        charset.update(characters)
        
        # Space-separated format
        tokenized_word = " ".join(characters)  # "_ h e l l o"
        vocabulary[tokenized_word] += 1
    
    return vocabulary, charset
```

**Why `_` marker?**
- Indicates word beginning
- Distinguishes "hello" vs "unhello"
- Critical for morpheme analysis

### 2. Pair Counting
```python
def _get_pair_counts(self):
    pair_counts = defaultdict(int)
    
    for tokenized_word, count in self.vocabulary.items():
        tokens = tokenized_word.split()  # "_ h e l l o" → ['_', 'h', 'e', 'l', 'l', 'o']
        
        # Count adjacent pairs
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])  # ('_', 'h'), ('h', 'e'), ...
            pair_counts[pair] += count  # Weighted by word frequency
    
    return pair_counts
```

### 3. Merge Operation
```python
def _merge_pair(self, pair):
    new_vocab = {}
    # Safe replacement with regex
    bigram = re.escape(' '.join(pair))  # "h e" → "h\\ e"
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word, count in self.vocabulary.items():
        # "_ h e l l o" → "_ he l l o"
        new_word = pattern.sub(''.join(pair), word)
        new_vocab[new_word] = count
        
    self.vocabulary = new_vocab
```

### 4. Main Training Loop
```python
def train(self, corpus):
    self.initialize_vocabulary(corpus)
    
    iteration = 0
    while len(self.tokens) < self.vocab_size:
        iteration += 1
        
        # 1. Count pairs
        pair_counts = self._get_pair_counts()
        if not pair_counts:
            break
            
        # 2. Find most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        
        # 3. Merge operation
        self._merge_pair(best_pair)
        
        # 4. Record keeping
        self.merges.append(best_pair)
        new_token = ''.join(best_pair)
        self.tokens.add(new_token)
```

---

## 🔧 Tokenization Process

### Post-Training Tokenization

```python
def tokenize(self, word):
    word_with_prefix = '_' + word
    
    # Split into characters, replace unknown with <UNK>
    tokens = [char if char in self.charset else self.unk_token 
              for char in word_with_prefix]
    
    # Apply merges in order
    while True:
        pairs_with_positions = []
        
        # Find available merges
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merge_map:
                merged_token, priority = self.merge_map[pair]
                pairs_with_positions.append((i, pair, merged_token, priority))
        
        if not pairs_with_positions:
            break
            
        # Apply highest priority merge
        pairs_with_positions.sort(key=lambda x: (x[3], x[0]))
        pos, _, merged_token, _ = pairs_with_positions[0]
        tokens[pos:pos+2] = [merged_token]
    
    return tokens
```

### Tokenization Example

```python
# Input: "modeling"
# Step 1: "_modeling" → ['_', 'm', 'o', 'd', 'e', 'l', 'i', 'n', 'g']

# Merge 1: ('_', 'm') → '_m'
# ['_m', 'o', 'd', 'e', 'l', 'i', 'n', 'g']

# Merge 2: ('_m', 'o') → '_mo'  
# ['_mo', 'd', 'e', 'l', 'i', 'n', 'g']

# Merge 3: ('_mo', 'd') → '_mod'
# ['_mod', 'e', 'l', 'i', 'n', 'g']

# ... continues ...

# Final: ['_model', 'ing']
```

---

## 📊 Results Analysis

### Real Output Analysis:
```
Test sentence: "Let's proceed to the language modeling part."

'Let's' → ['_', '<UNK>', 'et', "'", 's']
'proceed' → ['_proceed']  
'to' → ['_to']
'the' → ['_the']
'language' → ['_language']
'modeling' → ['_model', 'ing']
'part.' → ['_part', '.']
```

### Analysis:

#### ✅ Successful Tokenizations:
1. **`'proceed' → ['_proceed']`**: Learned as complete word
2. **`'modeling' → ['_model', 'ing']`**: Successful morpheme segmentation
3. **`'part.' → ['_part', '.']`**: Punctuation separation

#### ⚠️ Interesting Cases:
1. **`'Let's' → ['_', '<UNK>', 'et', "'", 's']`**:
   - `L` became `<UNK>` (uppercase rare in training)
   - Apostrophe correctly separated

#### 📈 What BPE Learned:
- **Frequent words**: `_the`, `_to`, `_proceed`
- **Morphemes**: `_model` + `ing`
- **Prefixes**: `_` marker effective
- **Punctuation**: Proper separation

---

## 🚀 Practical Application

### Running the Code:

```bash
# 1. Save and run the file
python bpe_tokenizer.py

# 2. Output:
=== BPE Tokenizer Training Demo ===

1. Preparing training data...        # Downloads Penn Treebank
2. Collecting corpus...              # Collects 500K words  
3. Training BPE tokenizer...         # Learns 4,951 merges
4. Saving tokenizer...               # Saves model
5. Testing tokenization...           # Tests on sample sentence
6. Testing save/load...              # Tests I/O functionality
```

### Test with Your Own Text:

```python
# Load your tokenizer
tokenizer = BPETokenizer()
tokenizer.load("bpe_tokenizer.pkl")

# Test it
test_words = ["happiness", "unhappy", "preprocessing", "artificial"]
for word in test_words:
    tokens = tokenizer.tokenize(word)
    print(f"{word} → {tokens}")
```

### Expected Outputs:
```python
happiness → ['_happy', 'ness']
unhappy → ['_un', 'happy'] 
preprocessing → ['_pre', 'process', 'ing']
artificial → ['_art', 'if', 'icial']
```

---

## 🎓 Advanced Topics

### 1. Hyperparameter Tuning

#### Vocabulary Size Impact:
```python
# Small vocab (1K): More subwords, less coverage
# Large vocab (50K): Fewer subwords, more coverage

vocab_size = 1000   # Aggressive subwording
vocab_size = 10000  # Balanced 
vocab_size = 50000  # Conservative subwording
```

### 2. Preprocessing Strategies

#### Case Handling:
```python
# Option 1: Lowercase everything
text = text.lower()

# Option 2: Keep case, handle separately  
# Special tokens for uppercase letters
```

#### Special Tokens:
```python
special_tokens = {
    "<UNK>": "Unknown characters",
    "<PAD>": "Padding", 
    "<START>": "Sequence start",
    "<END>": "Sequence end"
}
```

### 3. Evaluation Metrics

#### Coverage Analysis:
```python
def calculate_coverage(tokenizer, test_corpus):
    total_chars = 0
    unk_chars = 0
    
    for word in test_corpus:
        tokens = tokenizer.tokenize(word)
        for token in tokens:
            total_chars += len(token.replace('_', ''))
            if token == '<UNK>':
                unk_chars += 1
                
    coverage = (total_chars - unk_chars) / total_chars
    return coverage
```

#### Compression Ratio:
```python
def compression_ratio(original_text, tokenized_text):
    original_length = len(original_text.split())
    tokenized_length = sum(len(tokens) for tokens in tokenized_text)
    return tokenized_length / original_length
```

### 4. BPE Variants

#### **SentencePiece**:
- Developed by Google
- Unicode normalization
- Handles sentence boundaries

#### **WordPiece**:
- Used in BERT
- Likelihood-based merging
- `##` prefix for continuation

#### **Unigram Language Model**:
- Alternative in SentencePiece
- Probabilistic approach
- EM algorithm based

---

## 🔍 Debugging and Optimization

### Common Issues:

#### 1. Memory Issues:
```python
# Use streaming for large corpora
def create_word_generator(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            for word in line.split():
                yield word
```

#### 2. Speed Optimization:
```python
# Pre-compute merge map
def _build_merge_map(self):
    self.merge_map = {}
    for i, (left, right) in enumerate(self.merges):
        merged_token = left + right
        self.merge_map[(left, right)] = (merged_token, i)
```

#### 3. Regex Performance:
```python
# Compile regex patterns once
pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
# Reuse compiled pattern
```

---

## 🏆 Training Results Explanation

### Real Training Output Analysis:

```
Starting with 9,743 unique words and 49 characters
Target vocabulary size: 5,000

Iteration 100: Merged '_' + 'r' -> '_r'
Iteration 500: Merged '_am' + 'eric' -> '_americ'
Iteration 1000: Merged 'v' + 'ing' -> 'ving'
Iteration 2000: Merged '_co' + 'up' -> '_coup'
Iteration 4900: Merged '_account' + 'ed' -> '_accounted'

BPE training completed with 4,951 merges
Final vocabulary size: 5,000
```

### What This Shows:

#### **Evolution Pattern**:
- **Early iterations**: Basic character combinations (`'_' + 'r'`)
- **Mid iterations**: Common morphemes (`'v' + 'ing'`)
- **Late iterations**: Complete words (`'_account' + 'ed'`)

#### **Linguistic Insights**:
- **Prefix recognition**: `_americ` (American-related words)
- **Suffix learning**: `ving` (verb endings)
- **Word formation**: `_accounted` (complete inflected forms)

#### **Efficiency**:
- Started with 49 character vocabulary
- Reached 5,000 tokens with 4,951 merges
- Each merge adds exactly one new token

---

## 🎯 Summary

### BPE Advantages:
✅ **OOV handling**: Robust for unknown words  
✅ **Morphological awareness**: Separates roots and affixes  
✅ **Language agnostic**: Works across languages  
✅ **Configurable**: Adjustable vocabulary size  
✅ **Data-driven**: Learns from corpus patterns  

### Disadvantages:
❌ **Training time**: Slow for large corpora  
❌ **Ambiguity**: Multiple segmentations possible  
❌ **Domain dependency**: Different domains need different vocabularies  

### When to Use:
- 🔤 **Morphologically rich languages** (Turkish, German)
- 🌍 **Multilingual models** 
- 📚 **Domain-specific applications**
- 🤖 **Modern transformer models**

### Alternatives:
- **Word-level**: For simple applications
- **Character-level**: When very small vocabulary needed  
- **SentencePiece**: For production systems
- **WordPiece**: For BERT-style models

With this guide, you've learned BPE tokenization both theoretically and practically. You're ready to use it in your own projects! 🚀

---

## 📚 Additional Resources

1. **Original Paper**: Sennrich et al. (2016) - "Neural Machine Translation of Rare Words with Subword Units"
2. **Hugging Face Tokenizers**: Production-ready implementations
3. **SentencePiece**: Google's implementation
4. **OpenAI GPT**: BPE usage examples
5. **BERT Paper**: WordPiece variant explanation

**Happy tokenizing!** 🎉