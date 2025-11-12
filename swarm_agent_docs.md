# Swarm Agent Documentation

## Overview

The Swarm Agent is a powerful tool designed to query multiple AI models in parallel, providing a diverse range of perspectives on any given prompt. It is an ideal solution for tasks that benefit from a variety of AI viewpoints, such as making trading decisions, validating strategies, or assessing risks. The agent not only returns individual responses from each model but also generates a consensus summary, which is a three-sentence synthesis of all the responses, created by a separate AI model.

### Key Features:

- **Parallel AI Queries:** The agent can query up to six AI models simultaneously, including Claude 4.5 Sonnet, GPT-5, Gemini 2.5 Flash, Grok-4 Fast, DeepSeek Chat, and DeepSeek-R1.
- **Consensus Summary:** A standout feature of the Swarm Agent is its ability to generate a consensus summary of all the AI responses, providing a quick and easy way to understand the collective opinion of the models.
- **Easy to Use:** The agent can be run directly from the command line for interactive use or imported into other agents for programmatic access.
- **Flexible Configuration:** The `swarm_agent.py` file can be easily edited to enable or disable specific models, customize the consensus reviewer prompt, and adjust other parameters.
- **Detailed Response Structure:** The agent returns a clean JSON object containing the timestamp of the query, the original prompt, the consensus summary, a mapping of the AI models, the individual responses, and metadata about the query.
- **Automatic Data Logging:** All swarm queries are automatically saved to a JSON file for later review and analysis.

## Getting Started

Using the Swarm Agent is straightforward. You can either run it directly from the command line or import it into your own Python scripts.

### Method 1: Interactive Use

To use the Swarm Agent interactively, simply run the `swarm_agent.py` script from your terminal:

```bash
python src/agents/swarm_agent.py
```

The script will then prompt you to enter your question:

```
💭 What would you like to ask the AI swarm?
🌙 Prompt > [type your question here]
```

After you enter your prompt, the agent will query the active AI models and display their individual responses, along with the consensus summary.

### Method 2: Programmatic Use

To use the Swarm Agent in your own Python scripts, you can import the `SwarmAgent` class and use its `query` method:

```python
from src.agents.swarm_agent import SwarmAgent

# Initialize the SwarmAgent
swarm = SwarmAgent()

# Query the AI models
result = swarm.query("Should I buy Bitcoin at $100k?")

# Access the consensus summary
consensus = result["consensus_summary"]
print(consensus)

# Access the individual model responses
for provider, data in result["responses"].items():
    if data["success"]:
        print(f"{provider}: {data['response']}")

# Check the model mapping
for ai_num, provider in result["model_mapping"].items():
    print(f"{ai_num} = {provider}")
```

## Configuration

The Swarm Agent can be customized by editing the `swarm_agent.py` file.

- **Enable/Disable Models:** You can enable or disable specific models by setting the first element of the tuple to `True` or `False` in the `SWARM_MODELS` dictionary.
- **Consensus Reviewer:** The `CONSENSUS_REVIEWER_MODEL` variable specifies which AI model to use for generating the consensus summary.
- **Consensus Prompt:** The `CONSENSUS_REVIEWER_PROMPT` variable allows you to customize the prompt that is sent to the consensus reviewer.
- **Adjust Parameters:** You can also adjust the `DEFAULT_TEMPERATURE` and `DEFAULT_MAX_TOKENS` variables to control the creativity and length of the AI responses.

## Output

All swarm queries are automatically saved to a JSON file in the `src/data/swarm_agent/` directory. The filename is in the format `swarm_result_YYYYMMDD_HHMMSS.json`.
