"""
Byte Pair Encoding (BPE) Tokenizer Implementation

A comprehensive implementation of BPE tokenization for natural language processing.
Supports training, saving, loading, and tokenizing text using learned merges.
"""

import os
import urllib.request
import tarfile
import pickle
import re
import time
from typing import Dict, List, Tuple, Set, Iterator, Optional, DefaultDict
from collections import defaultdict


class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer class for training and tokenizing text.
    
    This implementation includes:
    - Training BPE from corpus
    - Fast tokenization with merge maps
    - Saving/loading tokenizer state
    - Handling unknown characters
    """
    
    def __init__(self, vocab_size: int = 5000, unk_token: str = "<UNK>"):
        """
        Initialize BPE tokenizer with configuration.
        
        Args:
            vocab_size: Target vocabulary size
            unk_token: Token to use for unknown characters
        """
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.merges: List[Tuple[str, str]] = []
        self.charset: Set[str] = set()
        self.tokens: Set[str] = set()
        self.merge_map: Dict[Tuple[str, str], Tuple[str, int]] = {}
        self.vocabulary: Dict[str, int] = {}
        
    def initialize_vocabulary(self, corpus: Iterator[str]) -> None:
        """
        Create initial vocabulary from corpus by splitting words into characters.
        
        Args:
            corpus: Iterator of words to process
        """
        vocabulary = defaultdict(int)
        charset = set()

        print("Initializing vocabulary from corpus...")
        for word in corpus:
            # Add word boundary marker and split into characters
            word_with_marker = '_' + word
            characters = list(word_with_marker)
            charset.update(characters)
            tokenized_word = " ".join(characters)
            vocabulary[tokenized_word] += 1

        self.vocabulary = dict(vocabulary)
        self.charset = charset
        self.tokens = set(charset)
        print(f"Initialized with {len(self.vocabulary)} unique words and {len(self.charset)} characters")
    
    def _get_pair_counts(self) -> DefaultDict[Tuple[str, str], int]:
        """
        Count frequencies of adjacent symbol pairs in the vocabulary.
        
        Returns:
            Dictionary mapping token pairs to their frequency counts
        """
        pair_counts = defaultdict(int)
        for tokenized_word, count in self.vocabulary.items():
            tokens = tokenized_word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += count
        return pair_counts
    
    def _merge_pair(self, pair: Tuple[str, str]) -> None:
        """
        Merge all occurrences of a specific symbol pair in the vocabulary.
        
        Args:
            pair: Tuple of tokens to merge
        """
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word, count in self.vocabulary.items():
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = count
            
        self.vocabulary = new_vocab
    
    def train(self, corpus: Iterator[str]) -> None:
        """
        Train BPE tokenizer on the provided corpus.
        
        Args:
            corpus: Iterator of words to learn from
        """
        print("Starting BPE training...")
        self.initialize_vocabulary(corpus)
        
        self.merges = []
        iteration = 0
        
        print(f"Target vocabulary size: {self.vocab_size}")
        
        while len(self.tokens) < self.vocab_size:
            iteration += 1
            pair_counts = self._get_pair_counts()
            
            if not pair_counts:
                print("No more pairs to merge")
                break
                
            best_pair = max(pair_counts, key=pair_counts.get)
            self._merge_pair(best_pair)
            
            self.merges.append(best_pair)
            new_token = ''.join(best_pair)
            self.tokens.add(new_token)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Merged '{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}'")
                print(f"Current vocabulary size: {len(self.tokens)}")
        
        # Build merge map for fast tokenization
        self._build_merge_map()
        print(f"BPE training completed with {len(self.merges)} merges")
    
    def _build_merge_map(self) -> None:
        """Build mapping from token pairs to their merged forms with priorities."""
        self.merge_map = {}
        for i, (left, right) in enumerate(self.merges):
            merged_token = left + right
            self.merge_map[(left, right)] = (merged_token, i)
    
    def tokenize(self, word: str) -> List[str]:
        """
        Tokenize a single word using learned BPE merges.
        
        Args:
            word: Word to tokenize
            
        Returns:
            List of tokens for the word
        """
        if not self.merge_map:
            self._build_merge_map()
            
        word_with_prefix = '_' + word
        
        # Handle words that exist as-is in vocabulary
        if word_with_prefix in self.vocabulary:
            return [word_with_prefix]

        # Initialize with characters, replacing unknown ones
        tokens = [char if char in self.charset else self.unk_token 
                 for char in word_with_prefix]

        # Apply merges iteratively
        while True:
            pairs_with_positions = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_map:
                    merged_token, merge_priority = self.merge_map[pair]
                    pairs_with_positions.append((i, pair, merged_token, merge_priority))

            if not pairs_with_positions:
                break

            # Sort by merge priority and position for consistency
            pairs_with_positions.sort(key=lambda x: (x[3], x[0]))
            pos, _, merged_token, _ = pairs_with_positions[0]
            tokens[pos:pos+2] = [merged_token]

        return tokens
    
    def tokenize_text(self, text: str) -> List[List[str]]:
        """
        Tokenize entire text by splitting into words and tokenizing each.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokenized words
        """
        return [self.tokenize(word) for word in text.split()]
    
    def save(self, filename: str = "bpe_tokenizer.pkl") -> None:
        """
        Save tokenizer state to file.
        
        Args:
            filename: Path to save tokenizer
        """
        tokenizer_state = {
            "vocab_size": self.vocab_size,
            "unk_token": self.unk_token,
            "merges": self.merges,
            "charset": self.charset,
            "tokens": self.tokens,
            "vocabulary": self.vocabulary
        }
        
        with open(filename, "wb") as f:
            pickle.dump(tokenizer_state, f)
        print(f"Tokenizer saved to {filename}")
    
    def load(self, filename: str = "bpe_tokenizer.pkl") -> None:
        """
        Load tokenizer state from file.
        
        Args:
            filename: Path to load tokenizer from
        """
        with open(filename, "rb") as f:
            tokenizer_state = pickle.load(f)
        
        self.vocab_size = tokenizer_state["vocab_size"]
        self.unk_token = tokenizer_state["unk_token"]
        self.merges = tokenizer_state["merges"]
        self.charset = tokenizer_state["charset"]
        self.tokens = tokenizer_state["tokens"]
        self.vocabulary = tokenizer_state["vocabulary"]
        
        # Rebuild merge map
        self._build_merge_map()
        print(f"Tokenizer loaded from {filename}")


class DataDownloader:
    """Utility class for downloading and preparing training data."""
    
    @staticmethod
    def _progress_hook(count: int, block_size: int, total_size: int) -> None:
        """Progress hook for urllib download."""
        downloaded = count * block_size
        percent = downloaded * 100 / (total_size or 1)
        mb_downloaded = downloaded // (1024 * 1024)
        print(f"\rDownloading... {percent:5.1f}% ({mb_downloaded} MB)", end="", flush=True)
    
    @staticmethod
    def download_file(url: str, filename: str) -> None:
        """
        Download file from URL if it doesn't exist locally.
        
        Args:
            url: URL to download from
            filename: Local filename to save to
        """
        if not os.path.exists(filename):
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, filename, DataDownloader._progress_hook)
            print("\nDownload completed.")
        else:
            print(f"{filename} already exists.")
    
    @staticmethod
    def _is_within_directory(directory: str, target: str) -> bool:
        """Security check to prevent path traversal attacks."""
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        prefix = os.path.commonprefix([abs_directory, abs_target])
        return prefix == abs_directory
    
    @staticmethod
    def safe_extract_tar(tar_file: str, required_files: List[str]) -> None:
        """
        Safely extract specific files from tar archive.
        
        Args:
            tar_file: Path to tar archive
            required_files: List of files to extract
        """
        with tarfile.open(tar_file, "r:gz") as tar:
            # Security check
            for member in tar.getmembers():
                if not DataDownloader._is_within_directory('.', member.name):
                    raise Exception("Attempted path traversal in tar file")
            
            # Extract required files
            for member in tar.getmembers():
                if any(member.name.endswith(file) for file in required_files):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, '.')
                    print(f"Extracted {member.name}")
    
    @staticmethod
    def create_word_generator(filepath: str) -> Iterator[str]:
        """
        Create generator that yields words from text file.
        
        Args:
            filepath: Path to text file
            
        Returns:
            Generator yielding individual words
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.split():
                    yield word
    
    @staticmethod
    def prepare_ptb_data() -> Iterator[str]:
        """
        Download and prepare Penn Treebank data from GitHub.
        
        Returns:
            Generator yielding words from training data
        """
        # Try to download from the original GitHub repo
        urls = {
            "train.txt": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
            "test.txt": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
            "valid.txt": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt"
        }
        
        for filename, url in urls.items():
            if not os.path.exists(filename):
                try:
                    print(f"Downloading {filename}...")
                    DataDownloader.download_file(url, filename)
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")
                    # Create a small sample file for demonstration
                    if filename == "train.txt":
                        DataDownloader._create_sample_data(filename)
                        break
            else:
                print(f"{filename} already exists.")
        
        # Check if we have training data
        if not os.path.exists("train.txt"):
            print("Creating sample training data for demonstration...")
            DataDownloader._create_sample_data("train.txt")

        return DataDownloader.create_word_generator("train.txt")
    
    @staticmethod
    def _create_sample_data(filename: str) -> None:
        """Create a sample text file for demonstration purposes."""
        sample_text = """
        the quick brown fox jumps over the lazy dog
        this is a sample text for training the tokenizer
        byte pair encoding is a subword tokenization algorithm
        it merges the most frequent pairs of characters or subwords
        the algorithm starts with individual characters as tokens
        then iteratively merges the most common adjacent pairs
        this process continues until the desired vocabulary size is reached
        the result is a vocabulary that can handle out of vocabulary words
        by breaking them down into known subword units
        this approach is widely used in natural language processing
        especially in neural machine translation and language modeling
        the tokenizer learns to represent common words as single tokens
        while rare words are split into multiple subword tokens
        this balances vocabulary size with representation quality
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Write the sample text multiple times to have enough data
            for _ in range(100):
                f.write(sample_text)
        
        print(f"Created sample data file: {filename}")


def main():
    """Main function demonstrating BPE tokenizer usage."""
    
    # Configuration
    VOCAB_SIZE = 5000
    MAX_CORPUS_SIZE = 500000
    
    print("=== BPE Tokenizer Training Demo ===\n")
    
    # Prepare data
    print("1. Preparing training data...")
    word_generator = DataDownloader.prepare_ptb_data()
    
    # Collect corpus
    print(f"\n2. Collecting corpus (max {MAX_CORPUS_SIZE:,} words)...")
    corpus = []
    for word in word_generator:
        corpus.append(word)
        if len(corpus) >= MAX_CORPUS_SIZE:
            break
    
    print(f"Collected {len(corpus):,} words for training")
    
    # Train tokenizer
    print(f"\n3. Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(iter(corpus))
    
    # Save tokenizer
    print(f"\n4. Saving tokenizer...")
    tokenizer.save("bpe_tokenizer.pkl")
    
    # Test tokenization
    print(f"\n5. Testing tokenization...")
    test_sentence = "Let's proceed to the language modeling part."
    
    # Time tokenization
    start_time = time.time()
    tokenized_words = tokenizer.tokenize_text(test_sentence)
    elapsed = time.time() - start_time
    
    print(f"\nTest sentence: '{test_sentence}'")
    print("Tokenization results:")
    for word, tokens in zip(test_sentence.split(), tokenized_words):
        print(f"  '{word}' -> {tokens}")
    
    print(f"\nTokenization time: {elapsed:.4f} seconds")
    print(f"Final vocabulary size: {len(tokenizer.tokens):,}")
    print(f"Number of merges learned: {len(tokenizer.merges):,}")
    
    # Test loading
    print(f"\n6. Testing save/load functionality...")
    new_tokenizer = BPETokenizer()
    new_tokenizer.load("bpe_tokenizer.pkl")
    
    # Verify loaded tokenizer works the same
    loaded_tokens = new_tokenizer.tokenize_text(test_sentence)
    print("Loaded tokenizer produces same results:", tokenized_words == loaded_tokens)
    
    print(f"\n=== Training completed successfully! ===")


if __name__ == "__main__":

    main()




"""
=== BPE Tokenizer Training Demo ===

1. Preparing training data...
Downloading train.txt...
Downloading from https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt...      
Downloading... 100.0% (4 MB)
Download completed.
Downloading test.txt...
Downloading from https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt...       
Downloading... 100.1% (0 MB)
Download completed.
Downloading valid.txt...
Downloading from https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt...      
Downloading... 100.4% (0 MB)
Download completed.

2. Collecting corpus (max 500,000 words)...
Collected 500,000 words for training

3. Training BPE tokenizer...
Starting BPE training...
Initializing vocabulary from corpus...
Initialized with 9743 unique words and 49 characters
Target vocabulary size: 5000
Iteration 100: Merged '_' + 'r' -> '_r'
Current vocabulary size: 149
Iteration 200: Merged 'p' + 'p' -> 'pp'
Current vocabulary size: 249
Iteration 300: Merged '_wh' + 'o' -> '_who'
Current vocabulary size: 349
Iteration 400: Merged '_p' + 'o' -> '_po'
Current vocabulary size: 449
Iteration 500: Merged '_am' + 'eric' -> '_americ'
Current vocabulary size: 549
Iteration 600: Merged '_un' + 'it' -> '_unit'
Current vocabulary size: 649
Iteration 700: Merged '_execut' + 'ive' -> '_executive'
Current vocabulary size: 749
Iteration 800: Merged '_f' + 'our' -> '_four'
Current vocabulary size: 849
Iteration 900: Merged '_ne' + 'ed' -> '_need'
Current vocabulary size: 949
Iteration 1000: Merged 'v' + 'ing' -> 'ving'
Current vocabulary size: 1049
Iteration 1100: Merged '_bel' + 'ie' -> '_belie'
Current vocabulary size: 1149
Iteration 1200: Merged '_lond' + 'on' -> '_london'
Current vocabulary size: 1249
Iteration 1300: Merged '_in' + 'fl' -> '_infl'
Current vocabulary size: 1349
Iteration 1400: Merged '_or' + 'der' -> '_order'
Current vocabulary size: 1449
Iteration 1500: Merged '_sh' + 'ar' -> '_shar'
Current vocabulary size: 1549
Iteration 1600: Merged '_l' + 'ater' -> '_later'
Current vocabulary size: 1649
Iteration 1700: Merged 'if' + 'ied' -> 'ified'
Current vocabulary size: 1749
Iteration 1800: Merged '_sug' + 'gest' -> '_suggest'
Current vocabulary size: 1849
Iteration 1900: Merged '_us' + 'ing' -> '_using'
Current vocabulary size: 1949
Iteration 2000: Merged '_co' + 'up' -> '_coup'
Current vocabulary size: 2049
Iteration 2100: Merged '_be' + 'h' -> '_beh'
Current vocabulary size: 2149
Iteration 2200: Merged '_d' + 'on' -> '_don'
Current vocabulary size: 2249
Iteration 2300: Merged 'ac' + 'ed' -> 'aced'
Current vocabulary size: 2349
Iteration 2400: Merged '_surve' + 'y' -> '_survey'
Current vocabulary size: 2449
Iteration 2500: Merged '_tr' + 'y' -> '_try'
Current vocabulary size: 2549
Iteration 2600: Merged '_sus' + 'p' -> '_susp'
Current vocabulary size: 2649
Iteration 2700: Merged '_us' + 'ually' -> '_usually'
Current vocabulary size: 2749
Iteration 2800: Merged '_respon' + 'se' -> '_response'
Current vocabulary size: 2849
Iteration 2900: Merged 'ut' + 'ton' -> 'utton'
Current vocabulary size: 2949
Iteration 3000: Merged 'en' + 'cies' -> 'encies'
Current vocabulary size: 3049
Iteration 3100: Merged '_sug' + 'ar' -> '_sugar'
Current vocabulary size: 3149
Iteration 3200: Merged 'ar' + 'ies' -> 'aries'
Current vocabulary size: 3249
Iteration 3300: Merged '_gold' + 'man' -> '_goldman'
Current vocabulary size: 3349
Iteration 3400: Merged '_s' + 'y' -> '_sy'
Current vocabulary size: 3449
Iteration 3500: Merged '_h' + 'ands' -> '_hands'
Current vocabulary size: 3549
Iteration 3600: Merged '_unl' + 'ike' -> '_unlike'
Current vocabulary size: 3649
Iteration 3700: Merged '_cut' + 'ting' -> '_cutting'
Current vocabulary size: 3749
Iteration 3800: Merged '_sa' + 'w' -> '_saw'
Current vocabulary size: 3849
Iteration 3900: Merged '_cor' + 'ry' -> '_corry'
Current vocabulary size: 3949
Iteration 4000: Merged 'i' + 'os' -> 'ios'
Current vocabulary size: 4049
Iteration 4100: Merged '_suggest' + 'ed' -> '_suggested'
Current vocabulary size: 4149
Iteration 4200: Merged '_j' + 'ury' -> '_jury'
Current vocabulary size: 4249
Iteration 4300: Merged '_black' + 's' -> '_blacks'
Current vocabulary size: 4349
Iteration 4400: Merged '_circ' + 'um' -> '_circum'
Current vocabulary size: 4449
Iteration 4500: Merged '_arch' + 'it' -> '_archit'
Current vocabulary size: 4549
Iteration 4600: Merged '_widesp' + 'read' -> '_widespread'
Current vocabulary size: 4649
Iteration 4700: Merged '_read' + 'ing' -> '_reading'
Current vocabulary size: 4749
Iteration 4800: Merged '_reins' + 'urance' -> '_reinsurance'
Current vocabulary size: 4849
Iteration 4900: Merged '_account' + 'ed' -> '_accounted'
Current vocabulary size: 4949
BPE training completed with 4951 merges

4. Saving tokenizer...
Tokenizer saved to bpe_tokenizer.pkl

5. Testing tokenization...

Test sentence: 'Let's proceed to the language modeling part.'
Tokenization results:
  'Let's' -> ['_', '<UNK>', 'et', "'", 's']
  'proceed' -> ['_proceed']
  'to' -> ['_to']
  'the' -> ['_the']
  'language' -> ['_language']
  'modeling' -> ['_model', 'ing']
  'part.' -> ['_part', '.']

Tokenization time: 0.0001 seconds
Final vocabulary size: 5,000
Number of merges learned: 4,951

6. Testing save/load functionality...
Tokenizer loaded from bpe_tokenizer.pkl
Loaded tokenizer produces same results: True

=== Training completed successfully! ===


"""
