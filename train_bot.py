#!/usr/bin/env python3
"""
Training Seed Script - Optional
This script allows you to pre-train the bot with example messages
before deploying it to Discord.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discord_learning_bot import LearningEngine

def train_from_file(filepath: str):
    """Train the bot from a text file with one message per line"""
    print(f"Loading messages from {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            messages = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return
    
    print(f"Found {len(messages)} messages")
    
    # Initialize learning engine
    engine = LearningEngine()
    print(f"Starting training count: {engine.training_count}")
    print(f"Starting vocabulary size: {engine.tokenizer.get_vocab_size()}")
    print()
    
    # Train on each message
    print("Training...")
    for i, message in enumerate(messages, 1):
        engine.learn_from_message(message)
        
        # Show progress
        if i % 10 == 0:
            print(f"  Processed {i}/{len(messages)} messages... (vocab: {engine.tokenizer.get_vocab_size()})")
    
    # Final save
    engine.save()
    
    print()
    print("✅ Training complete!")
    print(f"Final training count: {engine.training_count}")
    print(f"Final vocabulary size: {engine.tokenizer.get_vocab_size()}")
    print(f"Top words: {', '.join(engine.get_stats()['top_words'][:10])}")
    print()
    print("You can now run the Discord bot and it will continue learning from this state.")


def train_from_examples():
    """Train from built-in example messages"""
    print("Training from built-in examples...")
    
    # Expanded example conversation dataset for better pretraining
    examples = [
        # Greetings and basic conversation
        "hello everyone",
        "hey whats up",
        "hi there",
        "hello",
        "hey",
        "hi",
        "good morning",
        "good afternoon",
        "good evening",
        "whats going on",
        "how are you",
        "how are you doing",
        "im good thanks",
        "pretty good",
        "doing well",
        "not much you",
        "just chilling",
        "just hanging out",
        "same here",
        "nice nice",
        "cool cool",
        "awesome",
        "nice",
        "cool",
        "sweet",
        
        # Questions and responses
        "what are you up to",
        "what are you doing",
        "whatcha doing",
        "nothing much",
        "not a lot",
        "just relaxing",
        "watching some videos",
        "playing games",
        "listening to music",
        "browsing the internet",
        "talking to friends",
        
        # Agreement and acknowledgment
        "yeah",
        "yep",
        "yes",
        "yeah totally",
        "i agree",
        "makes sense",
        "that makes sense",
        "true",
        "true that",
        "for real",
        "for sure",
        "definitely",
        "absolutely",
        "exactly",
        "right",
        "thats right",
        "you got it",
        "i know right",
        "lol",
        "haha",
        "lmao",
        "thats funny",
        "thats hilarious",
        
        # Gaming conversation
        "anyone want to play games",
        "what game",
        "want to play",
        "lets play",
        "sure lets do it",
        "im down",
        "im in",
        "count me in",
        "maybe minecraft",
        "how about fortnite",
        "want to play valorant",
        "lets play league",
        "sounds good to me",
        "sounds fun",
        "that works",
        "good game",
        "gg",
        "nice play",
        "well played",
        "that was fun",
        "we should play again",
        "lets play again sometime",
        "im down for another round",
        
        # Food talk
        "pizza is the best",
        "i love pizza",
        "pizza time",
        "want to order pizza",
        "what kind of pizza",
        "pepperoni is my favorite",
        "i like cheese pizza",
        "what about supreme",
        "pineapple on pizza",
        "no pineapple on pizza",
        "pineapple belongs on pizza",
        "classic choice",
        "good choice",
        "cant go wrong with that",
        "im hungry",
        "same im starving",
        "lets get food",
        "what should we eat",
        "maybe tacos",
        "burgers sound good",
        "sushi is amazing",
        "i could go for chinese",
        
        # Casual reactions
        "wow",
        "woah",
        "oh wow",
        "really",
        "seriously",
        "no way",
        "for real",
        "are you serious",
        "bruh",
        "dude",
        "man",
        "omg",
        "wtf",
        "what the heck",
        "thats crazy",
        "thats insane",
        "thats wild",
        "interesting",
        "thats interesting",
        "hmm interesting",
        "i see",
        "i get it",
        "got it",
        "okay",
        "ok",
        "alright",
        "fair enough",
        
        # Goodbyes
        "see you later",
        "see you",
        "see ya",
        "catch you later",
        "talk to you later",
        "bye",
        "goodbye",
        "later",
        "peace",
        "have a good one",
        "have a good day",
        "take care",
        "good night",
        "gn",
        "night",
        
        # Asking for help or opinions
        "what do you think",
        "what are your thoughts",
        "any ideas",
        "suggestions",
        "can you help me",
        "need some help",
        "can someone help",
        "anyone know",
        "does anyone know",
        
        # Positive responses
        "thats great",
        "thats awesome",
        "thats cool",
        "love that",
        "love it",
        "i like that",
        "sounds great",
        "perfect",
        "excellent",
        "brilliant",
        
        # Negative/uncertain responses
        "not sure",
        "im not sure",
        "maybe",
        "i dont know",
        "idk",
        "no idea",
        "beats me",
        "who knows",
        "not really",
        "nah",
        "nope",
        "i dont think so",
        
        # Questions
        "why",
        "how",
        "when",
        "where",
        "who",
        "what",
        "why not",
        "how come",
        "what happened",
        "whats wrong",
        "you okay",
        "everything good",
        
        # Thanks and appreciation
        "thanks",
        "thank you",
        "thanks a lot",
        "appreciate it",
        "thanks dude",
        "ty",
        "thx",
        "no problem",
        "np",
        "no worries",
        "anytime",
        "youre welcome",
        "happy to help",
        "glad to help",
        
        # Random conversation starters
        "anyone here",
        "is anyone around",
        "quiet in here",
        "whats everyone doing",
        "hows everyone",
        "hows it going",
        "hows your day",
        "good day today",
        "had a good day",
        "today was fun",
        "today was crazy",
        
        # Weather and time
        "nice weather today",
        "its so hot",
        "its so cold",
        "love this weather",
        "hate this weather",
        "beautiful day",
        "what a day",
        "its getting late",
        "time flies",
        
        # Media discussion
        "anyone watching the game",
        "did you see that",
        "what are you watching",
        "watching anything good",
        "any good shows",
        "any recommendations",
        "just finished watching",
        "that was great",
        "that was terrible",
        "i loved it",
        "i hated it",
        
        # More natural conversation flow
        "honestly",
        "to be honest",
        "tbh",
        "actually",
        "basically",
        "pretty much",
        "kind of",
        "sort of",
        "i mean",
        "you know",
        "like",
        "just saying",
        "in my opinion",
        "i think",
        "i feel like",
        "seems like",
    ]
    
    engine = LearningEngine()
    print(f"Starting vocabulary size: {engine.tokenizer.get_vocab_size()}")
    
    for i, message in enumerate(examples, 1):
        engine.learn_from_message(message)
        if i % 25 == 0:
            print(f"  Processed {i}/{len(examples)} messages...")
    
    engine.save()
    
    print()
    print("✅ Training complete!")
    print(f"Final training count: {engine.training_count}")
    print(f"Final vocabulary size: {engine.tokenizer.get_vocab_size()}")
    print()
    
    # Test generation
    print("Testing generation...")
    for prompt in ["hello", "what are you doing", "pizza", "lets play", "thanks"]:
        response = engine.generate_response(context=prompt, temperature=1.2)
        print(f"  Input: {prompt} → Generated: {response}")
    
    print()
    print("The bot is now pre-trained! Run the Discord bot to continue learning.")


def interactive_training():
    """Interactive training mode"""
    print("Interactive Training Mode")
    print("Type messages to train the bot (type 'quit' to exit)")
    print("Type 'generate' to see a sample response")
    print()
    
    engine = LearningEngine()
    print(f"Current training count: {engine.training_count}")
    print(f"Current vocabulary size: {engine.tokenizer.get_vocab_size()}")
    print()
    
    while True:
        try:
            message = input("Message: ").strip()
            
            if message.lower() == 'quit':
                break
            
            if message.lower() == 'generate':
                response = engine.generate_response(temperature=1.2)
                print(f"  Bot says: {response}")
                print()
                continue
            
            if message:
                engine.learn_from_message(message)
                print(f"  Learned! (count: {engine.training_count}, vocab: {engine.tokenizer.get_vocab_size()})")
        
        except KeyboardInterrupt:
            break
    
    engine.save()
    print("\n✅ Training saved!")


def main():
    """Main entry point"""
    print("🤖 Self-Learning Bot - Training Script")
    print("=" * 50)
    print()
    
    if len(sys.argv) > 1:
        # Train from file
        filepath = sys.argv[1]
        train_from_file(filepath)
    else:
        # Show menu
        print("Choose an option:")
        print("  1. Train from example messages")
        print("  2. Train from file")
        print("  3. Interactive training")
        print("  4. Exit")
        print()
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            train_from_examples()
        elif choice == '2':
            filepath = input("Enter file path: ").strip()
            train_from_file(filepath)
        elif choice == '3':
            interactive_training()
        else:
            print("Exiting...")


if __name__ == "__main__":
    main()
