"""
Self-Learning Discord Bot with Tiny Neural Network
A Discord bot that learns from channel messages and evolves over time
"""

import discord
from discord import app_commands
from discord.ext import commands
import numpy as np
import json
import os
import pickle
import random
import re
from datetime import datetime
from collections import defaultdict, Counter
import asyncio
from typing import List, Dict, Optional, Tuple

# ============================================================================
# NEURAL NETWORK IMPLEMENTATION
# ============================================================================

class TinyLSTM:
    """
    Lightweight LSTM implementation for sequence learning.
    Designed to be CPU-friendly and memory-efficient.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 64):
        """
        Initialize the LSTM network.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden state
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Initialize embeddings
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # LSTM weights (input gate, forget gate, cell gate, output gate)
        self.W_i = np.random.randn(embedding_dim, hidden_dim) * 0.01
        self.U_i = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_i = np.zeros(hidden_dim)
        
        self.W_f = np.random.randn(embedding_dim, hidden_dim) * 0.01
        self.U_f = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_f = np.zeros(hidden_dim)
        
        self.W_c = np.random.randn(embedding_dim, hidden_dim) * 0.01
        self.U_c = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_c = np.zeros(hidden_dim)
        
        self.W_o = np.random.randn(embedding_dim, hidden_dim) * 0.01
        self.U_o = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_o = np.zeros(hidden_dim)
        
        # Output layer
        self.W_out = np.random.randn(hidden_dim, vocab_size) * 0.01
        self.b_out = np.zeros(vocab_size)
        
        # Learning rate - starts higher for faster early learning
        self.lr = 0.05
        self.training_steps = 0
        
    def sigmoid(self, x):
        """Numerically stable sigmoid"""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))
    
    def tanh(self, x):
        """Tanh activation"""
        return np.tanh(x)
    
    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, token_ids: List[int]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Forward pass through the LSTM.
        
        Args:
            token_ids: List of token indices
            
        Returns:
            output probabilities and cache for backprop
        """
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        cache = []
        
        for token_id in token_ids:
            # Get embedding
            x = self.embeddings[token_id]
            
            # Input gate
            i = self.sigmoid(np.dot(x, self.W_i) + np.dot(h, self.U_i) + self.b_i)
            
            # Forget gate
            f = self.sigmoid(np.dot(x, self.W_f) + np.dot(h, self.U_f) + self.b_f)
            
            # Cell gate
            c_tilde = self.tanh(np.dot(x, self.W_c) + np.dot(h, self.U_c) + self.b_c)
            
            # Update cell state
            c = f * c + i * c_tilde
            
            # Output gate
            o = self.sigmoid(np.dot(x, self.W_o) + np.dot(h, self.U_o) + self.b_o)
            
            # Update hidden state
            h = o * self.tanh(c)
            
            cache.append({
                'x': x, 'h_prev': h, 'c_prev': c,
                'i': i, 'f': f, 'c_tilde': c_tilde, 'o': o,
                'token_id': token_id
            })
        
        # Output layer
        logits = np.dot(h, self.W_out) + self.b_out
        probs = self.softmax(logits)
        
        return probs, cache
    
    def train_step(self, input_tokens: List[int], target_token: int) -> float:
        """
        Single training step with improved backprop.
        
        Args:
            input_tokens: Input sequence token IDs
            target_token: Target token ID to predict
            
        Returns:
            Loss value
        """
        # Forward pass
        probs, cache = self.forward(input_tokens)
        
        # Calculate loss (cross-entropy)
        loss = -np.log(probs[target_token] + 1e-8)
        
        # Adaptive learning rate (decreases over time for stability)
        self.training_steps += 1
        adaptive_lr = self.lr / (1.0 + self.training_steps / 1000.0)
        
        # Gradient for output
        d_logits = probs.copy()
        d_logits[target_token] -= 1
        
        # Get last hidden state
        if cache:
            h_last = cache[-1]['h_prev']
        else:
            h_last = np.zeros(self.hidden_dim)
        
        # Update output weights
        self.W_out -= adaptive_lr * np.outer(h_last, d_logits)
        self.b_out -= adaptive_lr * d_logits
        
        # Update embeddings and LSTM weights (improved backprop)
        d_h = np.dot(d_logits, self.W_out.T)
        
        # Backprop through last few LSTM steps for better learning
        for i, step in enumerate(reversed(cache[-10:])):
            token_id = step['token_id']
            
            # Update embeddings with stronger gradient
            gradient_strength = 0.3 * (1.0 - i / 10.0)  # Stronger for recent tokens
            d_embed = d_h[:self.embedding_dim] if len(d_h) >= self.embedding_dim else np.zeros(self.embedding_dim)
            self.embeddings[token_id] -= adaptive_lr * gradient_strength * d_embed[:self.embedding_dim]
            
            # Update LSTM output gate (helps with learning what to output)
            if i < 3:  # Only most recent for efficiency
                h_prev = step['h_prev']
                o = step['o']
                
                # Simple gradient updates for output gate
                d_o = d_h * np.tanh(step['c_prev']) * o * (1 - o)
                x = step['x']
                
                self.W_o -= adaptive_lr * 0.1 * np.outer(x, d_o)
                self.U_o -= adaptive_lr * 0.1 * np.outer(h_prev, d_o)
                self.b_o -= adaptive_lr * 0.1 * d_o
        
        return loss
    
    def generate(self, seed_tokens: List[int], max_length: int = 20, temperature: float = 1.0) -> List[int]:
        """
        Generate a sequence of tokens.
        
        Args:
            seed_tokens: Starting tokens
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated token IDs
        """
        generated = seed_tokens.copy()
        
        for _ in range(max_length):
            # Get probabilities for next token
            probs, _ = self.forward(generated[-10:])  # Use last 10 tokens as context
            
            # Apply temperature
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / np.sum(probs)
            
            # Sample next token
            next_token = np.random.choice(len(probs), p=probs)
            generated.append(next_token)
            
            # Stop if we generate end token (vocab_size - 1)
            if next_token == self.vocab_size - 1:
                break
        
        return generated[len(seed_tokens):]


# ============================================================================
# TOKENIZER AND VOCABULARY
# ============================================================================

class SimpleTokenizer:
    """Simple word-based tokenizer with vocabulary management"""
    
    def __init__(self):
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.next_id = 4
        self.word_freq = Counter()
        
    def preprocess(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Keep letters, numbers, spaces, and basic punctuation
        text = re.sub(r'[^a-z0-9\s!?.,\']', '', text)
        return text.strip()
    
    def tokenize(self, text: str, learn: bool = True) -> List[int]:
        """
        Tokenize text to IDs.
        
        Args:
            text: Input text
            learn: Whether to add new words to vocabulary
            
        Returns:
            List of token IDs
        """
        text = self.preprocess(text)
        words = text.split()
        
        token_ids = [self.word_to_id["<START>"]]
        
        for word in words:
            if word not in self.word_to_id:
                if learn:
                    self.word_to_id[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1
                    token_id = self.word_to_id[word]
                else:
                    token_id = self.word_to_id["<UNK>"]
            else:
                token_id = self.word_to_id[word]
            
            token_ids.append(token_id)
            if learn:
                self.word_freq[word] += 1
        
        token_ids.append(self.word_to_id["<END>"])
        return token_ids
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                word = self.id_to_word[token_id]
                if word not in ["<PAD>", "<UNK>", "<START>", "<END>"]:
                    words.append(word)
        return " ".join(words)
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size"""
        return len(self.word_to_id)


# ============================================================================
# LEARNING ENGINE
# ============================================================================

class LearningEngine:
    """Manages the learning process and model training"""
    
    def __init__(self, data_dir: str = "bot_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.tokenizer = SimpleTokenizer()
        self.model = None
        self.message_history = []
        self.training_count = 0
        
        # Load existing data if available
        self.load()
        
        # Initialize model if needed
        if self.model is None:
            vocab_size = max(100, self.tokenizer.get_vocab_size())
            
            # DYNAMIC SCALING: Continuous growth instead of thresholds
            # Formula: dimensions scale smoothly with vocabulary size
            # This ensures model capacity always matches vocabulary complexity
            
            # Base dimensions (for vocab ~100)
            base_embed = 32
            base_hidden = 64
            
            # Scale factor: grows logarithmically with vocab
            # log2(vocab/100) gives smooth scaling:
            # vocab=100 → scale=0 (base model)
            # vocab=200 → scale=1 (slightly bigger)
            # vocab=400 → scale=2
            # vocab=800 → scale=3
            # vocab=1600 → scale=4, etc.
            scale_factor = max(0, np.log2(vocab_size / 100))
            
            # Embedding dimension: grows by 8 per doubling of vocab
            # 100 words → 32 dim
            # 200 words → 40 dim
            # 400 words → 48 dim
            # 800 words → 56 dim
            # 1600 words → 64 dim
            # 3200 words → 72 dim
            # 6400 words → 80 dim
            embedding_dim = int(base_embed + scale_factor * 8)
            embedding_dim = min(128, embedding_dim)  # Cap at 128
            
            # Hidden dimension: grows by 16 per doubling of vocab
            # 100 words → 64 dim
            # 200 words → 80 dim
            # 400 words → 96 dim
            # 800 words → 112 dim
            # 1600 words → 128 dim
            # 3200 words → 144 dim
            # 6400 words → 160 dim
            # 12800 words → 176 dim
            hidden_dim = int(base_hidden + scale_factor * 16)
            hidden_dim = min(256, hidden_dim)  # Cap at 256
            
            print(f"Initializing model: vocab={vocab_size}, embed={embedding_dim}, hidden={hidden_dim}, scale={scale_factor:.2f}")
            
            self.model = TinyLSTM(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim
            )
    
    def learn_from_message(self, message: str):
        """
        Learn from a single message incrementally.
        
        Args:
            message: Message text to learn from
        """
        # Tokenize with learning enabled
        tokens = self.tokenizer.tokenize(message, learn=True)
        
        # Resize model if vocabulary grew
        if self.tokenizer.get_vocab_size() > self.model.vocab_size:
            self._resize_model()
        
        # Train on the message
        if len(tokens) > 2:  # Need at least START, word, END
            # Train to predict each token from previous context
            # Train on ALL subsequences for better learning (no random skip)
            for i in range(2, len(tokens)):
                input_seq = tokens[:i]
                target = tokens[i]
                
                loss = self.model.train_step(input_seq, target)
            
            self.training_count += 1
            
            # Store message in history (keep last 1000)
            self.message_history.append(message)
            if len(self.message_history) > 1000:
                self.message_history = self.message_history[-1000:]
            
            # Save periodically
            if self.training_count % 10 == 0:
                self.save()
    
    def generate_response(self, context: str = "", temperature: float = 1.0) -> str:
        """
        Generate a response based on context.
        
        Args:
            context: Optional context message
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        # Use context if provided, otherwise start fresh
        if context:
            seed_tokens = self.tokenizer.tokenize(context, learn=False)[-10:]
        else:
            seed_tokens = [self.tokenizer.word_to_id["<START>"]]
        
        # Adjust temperature based on training count (more confident over time)
        # Much less aggressive reduction so it stays creative
        adjusted_temp = temperature * max(0.7, 1.0 - (self.training_count / 2000))
        
        # Generate tokens
        generated_ids = self.model.generate(
            seed_tokens,
            max_length=random.randint(5, 20),  # Variable length
            temperature=adjusted_temp
        )
        
        # Convert to text
        response = self.tokenizer.detokenize(generated_ids)
        
        # Improved fallback for early stages
        if not response or len(response) < 2:
            if self.training_count < 20:
                # Very early - use simple responses
                return random.choice(["...", "hmm", "?", "ok", "hello", "hi", "yeah"])
            elif self.training_count < 50:
                # Echo random words from vocabulary
                common_words = [word for word, _ in self.tokenizer.word_freq.most_common(20)]
                if common_words:
                    num_words = random.randint(1, min(3, len(common_words)))
                    return " ".join(random.sample(common_words, num_words))
        
        return response if response else "..."
    
    def _resize_model(self):
        """Resize model when vocabulary grows - uses smooth dynamic scaling"""
        new_vocab_size = self.tokenizer.get_vocab_size()
        old_vocab_size = self.model.vocab_size
        
        if new_vocab_size > old_vocab_size:
            # Calculate what dimensions SHOULD be for new vocab size
            base_embed = 32
            base_hidden = 64
            scale_factor = max(0, np.log2(new_vocab_size / 100))
            
            new_embed = int(base_embed + scale_factor * 8)
            new_embed = min(128, new_embed)
            
            new_hidden = int(base_hidden + scale_factor * 16)
            new_hidden = min(256, new_hidden)
            
            old_embed = self.model.embedding_dim
            old_hidden = self.model.hidden_dim
            
            # Check if dimensions need to change (upgrade every ~200-400 words)
            # This creates smooth upgrades instead of sudden jumps
            needs_upgrade = (new_embed > old_embed) or (new_hidden > old_hidden)
            
            if needs_upgrade:
                print(f"🔧 Upgrading model: vocab {old_vocab_size}→{new_vocab_size}, dims {old_embed}/{old_hidden}→{new_embed}/{new_hidden}")
                
                # Save old embeddings for words we know
                old_embeddings = self.model.embeddings[:old_vocab_size, :old_embed]
                
                # Create new larger model
                new_model = TinyLSTM(
                    vocab_size=new_vocab_size,
                    embedding_dim=new_embed,
                    hidden_dim=new_hidden
                )
                
                # Copy over old embeddings (pad or truncate as needed)
                min_vocab = min(old_vocab_size, new_vocab_size)
                min_embed = min(old_embed, new_embed)
                new_model.embeddings[:min_vocab, :min_embed] = old_embeddings[:min_vocab, :min_embed]
                
                # Try to preserve LSTM weights if possible
                if old_hidden == new_hidden:
                    # Same hidden dim - can copy LSTM weights directly
                    # Need to handle embedding dim change in input weights
                    if old_embed == new_embed:
                        # Perfect match - copy everything
                        new_model.W_i = self.model.W_i
                        new_model.U_i = self.model.U_i
                        new_model.b_i = self.model.b_i
                        new_model.W_f = self.model.W_f
                        new_model.U_f = self.model.U_f
                        new_model.b_f = self.model.b_f
                        new_model.W_c = self.model.W_c
                        new_model.U_c = self.model.U_c
                        new_model.b_c = self.model.b_c
                        new_model.W_o = self.model.W_o
                        new_model.U_o = self.model.U_o
                        new_model.b_o = self.model.b_o
                        print("  ✓ Preserved all LSTM weights")
                    else:
                        # Embed changed but hidden same - partially preserve
                        # Copy what we can of the input weights
                        new_model.W_i[:min_embed, :] = self.model.W_i[:min_embed, :]
                        new_model.W_f[:min_embed, :] = self.model.W_f[:min_embed, :]
                        new_model.W_c[:min_embed, :] = self.model.W_c[:min_embed, :]
                        new_model.W_o[:min_embed, :] = self.model.W_o[:min_embed, :]
                        # Copy recurrent weights fully
                        new_model.U_i = self.model.U_i
                        new_model.U_f = self.model.U_f
                        new_model.U_c = self.model.U_c
                        new_model.U_o = self.model.U_o
                        # Copy biases
                        new_model.b_i = self.model.b_i
                        new_model.b_f = self.model.b_f
                        new_model.b_c = self.model.b_c
                        new_model.b_o = self.model.b_o
                        print("  ✓ Preserved recurrent LSTM weights, extended input weights")
                else:
                    # Hidden dim changed - fresh LSTM weights needed
                    print("  ⟳ Reset LSTM weights (hidden dim changed)")
                
                # Preserve training steps
                new_model.training_steps = self.model.training_steps
                
                # Replace model
                self.model = new_model
                return
            
            # No dimension change needed, just expand vocab size
            # Expand embeddings
            new_embeddings = np.random.randn(new_vocab_size, self.model.embedding_dim) * 0.01
            new_embeddings[:old_vocab_size] = self.model.embeddings
            self.model.embeddings = new_embeddings
            
            # Expand output layer
            new_W_out = np.random.randn(self.model.hidden_dim, new_vocab_size) * 0.01
            new_W_out[:, :old_vocab_size] = self.model.W_out
            self.model.W_out = new_W_out
            
            new_b_out = np.zeros(new_vocab_size)
            new_b_out[:old_vocab_size] = self.model.b_out
            self.model.b_out = new_b_out
            
            self.model.vocab_size = new_vocab_size
    
    def save(self):
        """Save model and tokenizer to disk"""
        # Save tokenizer
        tokenizer_path = os.path.join(self.data_dir, "tokenizer.pkl")
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save model weights
        model_path = os.path.join(self.data_dir, "model.npz")
        np.savez(
            model_path,
            embeddings=self.model.embeddings,
            W_i=self.model.W_i, U_i=self.model.U_i, b_i=self.model.b_i,
            W_f=self.model.W_f, U_f=self.model.U_f, b_f=self.model.b_f,
            W_c=self.model.W_c, U_c=self.model.U_c, b_c=self.model.b_c,
            W_o=self.model.W_o, U_o=self.model.U_o, b_o=self.model.b_o,
            W_out=self.model.W_out, b_out=self.model.b_out,
            vocab_size=self.model.vocab_size,
            embedding_dim=self.model.embedding_dim,
            hidden_dim=self.model.hidden_dim,
            training_steps=self.model.training_steps  # Save training steps too
        )
        
        # Save metadata
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'training_count': self.training_count,
                'message_history': self.message_history[-100:]  # Save last 100
            }, f)
    
    def load(self):
        """Load model and tokenizer from disk"""
        tokenizer_path = os.path.join(self.data_dir, "tokenizer.pkl")
        model_path = os.path.join(self.data_dir, "model.npz")
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        
        # Load tokenizer
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        
        # Load model
        if os.path.exists(model_path):
            data = np.load(model_path)
            
            self.model = TinyLSTM(
                vocab_size=int(data['vocab_size']),
                embedding_dim=int(data['embedding_dim']),
                hidden_dim=int(data['hidden_dim'])
            )
            
            self.model.embeddings = data['embeddings']
            self.model.W_i = data['W_i']
            self.model.U_i = data['U_i']
            self.model.b_i = data['b_i']
            self.model.W_f = data['W_f']
            self.model.U_f = data['U_f']
            self.model.b_f = data['b_f']
            self.model.W_c = data['W_c']
            self.model.U_c = data['U_c']
            self.model.b_c = data['b_c']
            self.model.W_o = data['W_o']
            self.model.U_o = data['U_o']
            self.model.b_o = data['b_o']
            self.model.W_out = data['W_out']
            self.model.b_out = data['b_out']
            
            # Load training steps if available (backward compatibility)
            if 'training_steps' in data:
                self.model.training_steps = int(data['training_steps'])
            
            print(f"Loaded model: vocab={self.model.vocab_size}, embed={self.model.embedding_dim}, hidden={self.model.hidden_dim}")
        
        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.training_count = metadata.get('training_count', 0)
                self.message_history = metadata.get('message_history', [])
    
    def get_stats(self) -> Dict:
        """Get learning statistics"""
        return {
            'training_count': self.training_count,
            'vocab_size': self.tokenizer.get_vocab_size(),
            'messages_stored': len(self.message_history),
            'top_words': [word for word, _ in self.tokenizer.word_freq.most_common(10)]
        }


# ============================================================================
# DISCORD BOT
# ============================================================================

class LearningBot(commands.Bot):
    """Discord bot with self-learning capabilities"""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        
        super().__init__(command_prefix="!", intents=intents)
        
        self.learning_engine = LearningEngine()
        self.config_path = "bot_data/config.json"
        self.load_config()
        
    def load_config(self):
        """Load bot configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.learning_channel_id = config.get('learning_channel_id')
                # Handle backward compatibility with old boolean setting
                old_standalone = config.get('respond_to_standalone', False)
                if isinstance(old_standalone, bool):
                    self.standalone_response_rate = 15.0 if old_standalone else 0.0
                else:
                    self.standalone_response_rate = config.get('standalone_response_rate', 0.0)
        else:
            self.learning_channel_id = None
            self.standalone_response_rate = 0.0
    
    def save_config(self):
        """Save bot configuration"""
        os.makedirs("bot_data", exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump({
                'learning_channel_id': self.learning_channel_id,
                'standalone_response_rate': self.standalone_response_rate
            }, f)
    
    async def setup_hook(self):
        """Setup hook called when bot is ready"""
        await self.tree.sync()
        print("Commands synced!")


# Initialize bot
bot = LearningBot()


# ============================================================================
# EVENT HANDLERS
# ============================================================================

@bot.event
async def on_ready():
    """Called when bot is ready"""
    print(f'{bot.user} has connected to Discord!')
    print(f'Training count: {bot.learning_engine.training_count}')
    print(f'Vocabulary size: {bot.learning_engine.tokenizer.get_vocab_size()}')
    if bot.learning_channel_id:
        print(f'Learning channel ID: {bot.learning_channel_id}')
    else:
        print('No learning channel configured. Use /config to set one.')


@bot.event
async def on_message(message: discord.Message):
    """Handle incoming messages"""
    # Ignore own messages
    if message.author == bot.user:
        return
    
    # Ignore bot messages (optional - comment out to learn from other bots)
    if message.author.bot:
        return
    
    # Learn from messages in learning channel
    if bot.learning_channel_id and message.channel.id == bot.learning_channel_id:
        if message.content and not message.content.startswith('/'):
            # Learn from this message
            bot.learning_engine.learn_from_message(message.content)
            print(f"Learned from message. Training count: {bot.learning_engine.training_count}")
    
    # Decide whether to respond
    should_respond = False
    context = ""
    
    # Check if bot was mentioned
    if bot.user in message.mentions:
        should_respond = True
        # Remove mention from context
        context = message.content.replace(f'<@{bot.user.id}>', '').strip()
    
    # Check if replying to bot's message
    elif message.reference:
        try:
            referenced_message = await message.channel.fetch_message(message.reference.message_id)
            if referenced_message.author == bot.user:
                should_respond = True
                context = message.content
        except:
            pass
    
    # Check if standalone responses are enabled in learning channel
    elif (bot.learning_channel_id and 
          message.channel.id == bot.learning_channel_id and 
          bot.standalone_response_rate > 0):
        # Respond based on configured percentage
        if random.random() < (bot.standalone_response_rate / 100.0):
            should_respond = True
            context = message.content
    
    # Generate and send response
    if should_respond:
        async with message.channel.typing():
            # Add slight delay to seem more natural
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Generate response
            response = bot.learning_engine.generate_response(
                context=context,
                temperature=1.2  # Slightly random
            )
            
            # Ensure response isn't empty
            if not response or len(response.strip()) < 1:
                response = "..."
            
            # Send response
            await message.reply(response)


# ============================================================================
# SLASH COMMANDS
# ============================================================================

@bot.tree.command(name="config", description="Configure the learning bot")
@app_commands.describe(
    learning_channel="The channel to learn from",
    standalone_response_rate="Percentage (0-100) of standalone messages to respond to in learning channel"
)
async def config(
    interaction: discord.Interaction,
    learning_channel: Optional[discord.TextChannel] = None,
    standalone_response_rate: Optional[float] = None
):
    """Configure bot settings"""
    changes = []
    
    if learning_channel:
        bot.learning_channel_id = learning_channel.id
        changes.append(f"Learning channel set to {learning_channel.mention}")
    
    if standalone_response_rate is not None:
        # Validate range
        if standalone_response_rate < 0 or standalone_response_rate > 100:
            await interaction.response.send_message(
                "❌ Standalone response rate must be between 0 and 100",
                ephemeral=True
            )
            return
        
        bot.standalone_response_rate = standalone_response_rate
        if standalone_response_rate == 0:
            changes.append("Standalone responses disabled")
        else:
            changes.append(f"Standalone response rate set to {standalone_response_rate}%")
    
    if changes:
        bot.save_config()
        await interaction.response.send_message(
            "Configuration updated:\n" + "\n".join(f"• {change}" for change in changes),
            ephemeral=True
        )
    else:
        # Show current config
        learning_channel_text = f"<#{bot.learning_channel_id}>" if bot.learning_channel_id else "Not set"
        standalone_text = f"{bot.standalone_response_rate}%" if bot.standalone_response_rate > 0 else "Disabled (0%)"
        await interaction.response.send_message(
            f"**Current Configuration:**\n"
            f"• Learning channel: {learning_channel_text}\n"
            f"• Standalone response rate: {standalone_text}",
            ephemeral=True
        )


@bot.tree.command(name="status", description="Show bot learning status")
async def status(interaction: discord.Interaction):
    """Show bot statistics"""
    stats = bot.learning_engine.get_stats()
    
    # Determine learning stage
    count = stats['training_count']
    if count < 10:
        stage = "🌱 Just born (gibberish phase)"
    elif count < 30:
        stage = "👶 Early learning (word discovery)"
    elif count < 70:
        stage = "🧒 Growing (basic phrases)"
    elif count < 150:
        stage = "🧑 Developing (context awareness)"
    else:
        stage = "🧠 Evolved (personality emerging)"
    
    embed = discord.Embed(
        title="🤖 Learning Bot Status",
        color=discord.Color.blue(),
        timestamp=datetime.utcnow()
    )
    
    embed.add_field(name="Learning Stage", value=stage, inline=False)
    embed.add_field(name="Messages Learned", value=f"{count:,}", inline=True)
    embed.add_field(name="Vocabulary Size", value=f"{stats['vocab_size']:,} words", inline=True)
    embed.add_field(name="Memory Size", value=f"{stats['messages_stored']} messages", inline=True)
    
    if stats['top_words']:
        top_words = ", ".join(stats['top_words'][:8])
        embed.add_field(name="Common Words", value=top_words, inline=False)
    
    learning_channel = f"<#{bot.learning_channel_id}>" if bot.learning_channel_id else "Not configured"
    embed.add_field(name="Learning Channel", value=learning_channel, inline=True)
    
    standalone_text = f"{bot.standalone_response_rate}%" if bot.standalone_response_rate > 0 else "Disabled"
    embed.add_field(name="Standalone Rate", value=standalone_text, inline=True)
    
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="reset", description="Reset the bot's learning (admin only)")
async def reset(interaction: discord.Interaction):
    """Reset bot learning"""
    # Check if user has admin permissions
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "❌ You need administrator permissions to reset the bot.",
            ephemeral=True
        )
        return
    
    # Confirm reset
    await interaction.response.send_message(
        "⚠️ Are you sure you want to reset all learning? This cannot be undone!\n"
        "Type `/confirm_reset` to proceed.",
        ephemeral=True
    )


@bot.tree.command(name="confirm_reset", description="Confirm learning reset (admin only)")
async def confirm_reset(interaction: discord.Interaction):
    """Confirm and execute reset"""
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "❌ You need administrator permissions to reset the bot.",
            ephemeral=True
        )
        return
    
    # Reset learning engine
    bot.learning_engine = LearningEngine()
    
    await interaction.response.send_message(
        "✅ Bot learning has been reset. Starting fresh!",
        ephemeral=True
    )


@bot.tree.command(name="generate", description="Generate a random response")
@app_commands.describe(context="Optional context for generation")
async def generate(interaction: discord.Interaction, context: Optional[str] = None):
    """Manually generate a response"""
    await interaction.response.defer()
    
    response = bot.learning_engine.generate_response(
        context=context or "",
        temperature=1.3
    )
    
    if not response:
        response = "..."
    
    await interaction.followup.send(f"Generated: {response}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    # Check for token
    token = os.getenv('DISCORD_BOT_TOKEN')
    
    if not token:
        print("❌ ERROR: DISCORD_BOT_TOKEN environment variable not set!")
        print("\nPlease set your Discord bot token:")
        print("  Linux/Mac: export DISCORD_BOT_TOKEN='your-token-here'")
        print("  Windows: set DISCORD_BOT_TOKEN=your-token-here")
        print("\nOr create a .env file with: DISCORD_BOT_TOKEN=your-token-here")
        return
    
    print("🚀 Starting Self-Learning Discord Bot...")
    print("=" * 50)
    
    try:
        bot.run(token)
    except discord.LoginFailure:
        print("❌ ERROR: Invalid Discord token!")
    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    main()
