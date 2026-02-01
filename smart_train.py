#!/usr/bin/env python3
"""
Smart Training Script - Convert Books/Text to Chat-Style Messages
Fixes the problem of training on non-conversational text
"""

import sys
import os
import re
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discord_learning_bot import LearningEngine


def extract_dialogue_from_text(text):
    """
    Extract dialogue/conversational parts from book text.
    Books have different structure than chat - this fixes it.
    """
    dialogue = []
    
    # Find quoted dialogue
    # Pattern: "Something someone said"
    quotes = re.findall(r'"([^"]{10,100})"', text)
    dialogue.extend(quotes)
    
    # Find sentences with conversational markers
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        # Keep sentences that look conversational
        if any(word in sentence.lower() for word in [
            'i ', 'you ', 'we ', 'they ', 'what', 'how', 'why', 'when',
            'said', 'asked', 'told', 'replied', 'yeah', 'yes', 'no'
        ]):
            if 10 < len(sentence) < 100:  # Good chat length
                dialogue.append(sentence)
    
    return dialogue


def convert_to_chat_style(text, max_messages=1000):
    """
    Convert book text into chat-style messages.
    Breaks into short sentences, filters out narrative.
    """
    print("Converting book text to chat-style messages...")
    
    chat_messages = []
    
    # First, try to extract dialogue
    dialogue = extract_dialogue_from_text(text)
    chat_messages.extend(dialogue)
    print(f"  Found {len(dialogue)} dialogue snippets")
    
    # Then, break remaining text into chat-sized chunks
    # Remove newlines and normalize spacing
    text = re.sub(r'\s+', ' ', text)
    
    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Skip if too short or too long for chat
        if len(sentence) < 10 or len(sentence) > 120:
            continue
        
        # Skip if it's too "book-like" (has fancy words)
        if any(word in sentence.lower() for word in [
            'therefore', 'furthermore', 'nevertheless', 'notwithstanding',
            'aforementioned', 'henceforth', 'wherein', 'thereby'
        ]):
            continue
        
        # Skip if no conversational words
        has_conversational = any(word in sentence.lower() for word in [
            'i ', 'you ', 'we ', 'my ', 'your ', 'our ', 'what', 'how', 'why',
            'is ', 'are ', 'was ', 'were ', 'can ', 'will ', 'would ', 'could'
        ])
        
        if has_conversational:
            chat_messages.append(sentence)
    
    # Deduplicate
    chat_messages = list(set(chat_messages))
    
    # Shuffle
    random.shuffle(chat_messages)
    
    # Limit to max_messages
    chat_messages = chat_messages[:max_messages]
    
    print(f"  Converted to {len(chat_messages)} chat-style messages")
    
    return chat_messages


def train_with_better_strategy(messages, batch_size=10):
    """
    Train with a smarter strategy:
    - Multiple epochs on small batches
    - Repetition for reinforcement
    - Progress monitoring
    """
    print("\n" + "=" * 60)
    print("Smart Training Strategy")
    print("=" * 60)
    
    engine = LearningEngine()
    initial_count = engine.training_count
    initial_vocab = engine.tokenizer.get_vocab_size()
    
    print(f"Starting state:")
    print(f"  Training count: {initial_count}")
    print(f"  Vocabulary: {initial_vocab}")
    print()
    
    # Train in multiple passes for better learning
    num_epochs = 3  # Repeat the dataset
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # Shuffle for each epoch
        shuffled = messages.copy()
        random.shuffle(shuffled)
        
        for i, message in enumerate(shuffled, 1):
            engine.learn_from_message(message)
            
            # Show progress every 50 messages
            if i % 50 == 0:
                vocab_now = engine.tokenizer.get_vocab_size()
                print(f"  {i}/{len(shuffled)} messages | vocab: {vocab_now} | training count: {engine.training_count}")
                
                # Test generation every 100
                if i % 100 == 0:
                    test_response = engine.generate_response(temperature=1.0)
                    print(f"    Test: '{test_response}'")
        
        print(f"  Epoch {epoch + 1} complete!")
        print()
    
    # Final save
    engine.save()
    
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final training count: {engine.training_count} (was {initial_count})")
    print(f"Final vocabulary: {engine.tokenizer.get_vocab_size()} (was {initial_vocab})")
    print()
    
    # Test with various prompts
    print("Testing generation with different prompts:")
    print("-" * 60)
    
    test_prompts = [
        "",  # Random
        "hello",
        "what",
        "how are you",
        "i think",
        "yes",
        "no",
        "the"
    ]
    
    for prompt in test_prompts:
        response = engine.generate_response(context=prompt, temperature=1.0)
        prompt_display = prompt if prompt else "(random)"
        print(f"  '{prompt_display}' → '{response}'")
    
    print()
    print("=" * 60)
    print("Bot is now trained!")
    print("Start the Discord bot to continue learning from chat.")
    print("=" * 60)


def train_from_book_file(filepath):
    """Train from a book file with smart conversion"""
    print(f"Loading book from: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return
    
    print(f"Loaded {len(text):,} characters")
    
    # Convert to chat-style
    chat_messages = convert_to_chat_style(text, max_messages=500)
    
    if len(chat_messages) < 50:
        print("⚠️  WARNING: Very few chat-style messages extracted!")
        print("This book might not be suitable for training.")
        print(f"Found only {len(chat_messages)} usable messages.")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Train with smart strategy
    train_with_better_strategy(chat_messages)


def create_synthetic_chat_dataset(size=500):
    """
    Create a synthetic chat dataset with natural patterns.
    Much better than book text for training!
    """
    print("Creating synthetic chat dataset...")
    
    # Templates for natural conversation
    greetings = ["hello", "hey", "hi", "sup", "yo", "hey there", "whats up", "howdy"]
    
    questions = [
        "what are you doing",
        "how are you",
        "whats going on",
        "you okay",
        "what do you think",
        "any ideas",
        "what happened",
        "where are you",
        "when did that happen",
    ]
    
    responses_positive = [
        "yeah", "yes", "yep", "sure", "definitely", "absolutely",
        "for sure", "i agree", "totally", "exactly", "right",
        "thats right", "makes sense", "good idea", "sounds good"
    ]
    
    responses_negative = [
        "no", "nah", "nope", "not really", "i dont think so",
        "maybe not", "probably not", "idk", "not sure"
    ]
    
    statements = [
        "i like this", "this is good", "thats cool", "nice",
        "that makes sense", "i see", "interesting", "got it",
        "i get it", "fair enough", "alright", "okay"
    ]
    
    reactions = [
        "wow", "woah", "oh wow", "really", "seriously",
        "no way", "omg", "lol", "haha", "lmao",
        "thats crazy", "thats wild", "bruh", "dude"
    ]
    
    # Combine all
    all_patterns = (
        greetings * 20 +
        questions * 15 +
        responses_positive * 15 +
        responses_negative * 10 +
        statements * 15 +
        reactions * 15
    )
    
    # Add variations
    messages = []
    for pattern in all_patterns:
        messages.append(pattern)
        # Add with punctuation
        if random.random() < 0.3:
            messages.append(pattern + "!")
        if random.random() < 0.2:
            messages.append(pattern + "?")
    
    # Add combinations
    for _ in range(100):
        msg = random.choice(greetings) + " " + random.choice(questions)
        messages.append(msg)
    
    for _ in range(100):
        msg = random.choice(responses_positive) + " " + random.choice(statements)
        messages.append(msg)
    
    # Shuffle and limit
    random.shuffle(messages)
    messages = messages[:size]
    
    print(f"Created {len(messages)} synthetic messages")
    return messages


def main():
    print("=" * 60)
    print("Smart Training Script for Discord Bot")
    print("=" * 60)
    print()
    print("⚠️  IMPORTANT: Books don't work well for chat training!")
    print("   Books have narrative structure, not conversation.")
    print("   This script will try to fix that.\n")
    
    if len(sys.argv) > 1:
        # Train from file
        filepath = sys.argv[1]
        train_from_book_file(filepath)
    else:
        # Interactive menu
        print("Choose training option:")
        print("  1. Synthetic chat dataset (RECOMMENDED - best for Discord)")
        print("  2. Convert book/text file to chat-style")
        print("  3. Original example messages (from train_bot.py)")
        print("  4. Exit")
        print()
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            print("\nHow many messages to generate?")
            size = input("Enter size (100-1000, recommended 500): ").strip()
            try:
                size = int(size)
                size = max(100, min(1000, size))
            except:
                size = 500
            
            messages = create_synthetic_chat_dataset(size)
            train_with_better_strategy(messages)
            
        elif choice == '2':
            filepath = input("Enter path to book/text file: ").strip()
            train_from_book_file(filepath)
            
        elif choice == '3':
            print("\nUsing original train_bot.py examples...")
            os.system("python3 train_bot.py")
            
        else:
            print("Exiting...")


if __name__ == "__main__":
    main()
