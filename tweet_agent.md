# Tweet Agent Documentation

## Overview

The Tweet Agent is a Python script that generates tweets from a given text input. It uses an AI model to create multiple tweets based on the content of the text, which can be provided as a string or read from a file. The agent is designed to be highly customizable, allowing you to specify the AI model, temperature, and maximum number of tokens to use for tweet generation.

### Key Features:

- **Text-to-Tweet Generation:** The agent can take a long piece of text and generate multiple tweets from it.
- **AI-Powered:** It uses an AI model to generate the tweets, with support for both DeepSeek and Claude.
- **Customizable AI Settings:** You can easily override the default AI model, temperature, and maximum number of tokens.
- **Flexible Input:** The agent can read text from a file or a string.
- **Chunking:** It can split large texts into smaller chunks to process them more efficiently.
- **Colored Output:** The generated tweets are printed to the console with colored backgrounds for easy readability.
- **File Output:** The tweets are saved to a text file with a timestamp in the filename.

## Getting Started

To use the Tweet Agent, you can either run it directly from the command line or import it into your own Python scripts.

### Method 1: Interactive Use

To use the Tweet Agent interactively, simply run the `tweet_agent.py` script from your terminal:

```bash
python src/agents/tweet_agent.py
```

The script will then generate tweets from the `og_tweet_text.txt` file and save them to a new text file in the `src/data/tweets` directory.

### Method 2: Programmatic Use

To use the Tweet Agent in your own Python scripts, you can import the `TweetAgent` class and use its `generate_tweets` method:

```python
from src/agents/tweet_agent import TweetAgent

# Initialize the TweetAgent
agent = TweetAgent()

# Provide the text to generate tweets from
text = "This is a long piece of text that I want to generate tweets from."

# Generate the tweets
tweets = agent.generate_tweets(text)

# Print the generated tweets
if tweets:
    for tweet in tweets:
        print(tweet)
```

## Configuration

The Tweet Agent can be configured by editing the `tweet_agent.py` file.

- **`MODEL_OVERRIDE`:** This variable allows you to override the default AI model. You can set it to "deepseek-chat", "deepseek-reasoner", or "0" to use the `AI_MODEL` setting from `config.py`.
- **`MAX_CHUNK_SIZE`:** This variable specifies the maximum number of characters per chunk.
- **`TWEETS_PER_CHUNK`:** This variable specifies the number of tweets to generate per chunk.
- **`USE_TEXT_FILE`:** This variable determines whether to use the `og_tweet_text.txt` file as input.
- **`OG_TWEET_FILE`:** This variable specifies the path to the input text file.
- **`AI_MODEL`:** This variable allows you to override the `AI_MODEL` setting from `config.py`.
- **`AI_TEMPERATURE`:** This variable allows you to override the `AI_TEMPERATURE` setting from `config.py`.
- **`AI_MAX_TOKENS`:** This variable allows you to override the `AI_MAX_TOKENS` setting from `config.py`.

## API Requirements

To use the Tweet Agent, you need to add the following API keys to your `.env` file:

- `OPENAI_KEY`
- `ANTHROPIC_KEY`
- `DEEPSEEK_KEY`
