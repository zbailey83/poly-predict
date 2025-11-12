# Polymarket Prediction Agent

## Overview

The Polymarket Prediction Agent is a sophisticated tool designed to monitor, analyze, and predict outcomes on the Polymarket prediction market. It operates in real-time by connecting to the Polymarket WebSocket, capturing significant trades, and leveraging a swarm of AI models to generate insightful predictions. The agent is equipped with features to filter markets, identify consensus among AI models, and save data for historical analysis.

### Key Features:

- **Real-time Monitoring:** Connects to Polymarket's WebSocket to watch live trades.
- **Configurable Trade Filtering:** By default, it tracks trades over $100, but this can be adjusted.
- **Smart Market Filtering:** Filters out markets related to crypto and sports, and those near resolution.
- **AI Swarm Analysis:** Utilizes a configurable swarm of 6 AI models to analyze markets in parallel.
- **Consensus Identification:** Employs a consensus mechanism to identify the top 5 market predictions with the strongest agreement among the AI models.
- **Clear Visual Cues:** Displays consensus picks with a distinct blue background for easy identification.
- **Comprehensive Data Logging:** Saves all market data, individual AI predictions, and consensus picks into three separate CSV files for thorough historical tracking and analysis.

## Quick Start

To get the Polymarket Prediction Agent up and running, follow these simple steps:

1. **Activate Environment:** Ensure you are in the correct Conda environment.
   ```bash
   conda activate tflow
   ```

2. **Run the Agent:** Execute the agent script from the command line.
   ```bash
   python src/agents/polymarket_agent.py
   ```

## Configuration

All settings for the Polymarket Prediction Agent can be easily customized at the top of the `polymarket_agent.py` file.

### Basic Settings

- `MIN_TRADE_SIZE_USD`: The minimum trade size in USD to track. Default is `100`.
- `LOOKBACK_HOURS`: The number of hours to look back for historical data upon startup. Default is `24`.
- `NEW_MARKETS_FOR_ANALYSIS`: The number of new markets to collect before triggering an AI analysis. Default is `25`.
- `ANALYSIS_CHECK_INTERVAL_SECONDS`: The interval in seconds at which the agent checks for new markets to analyze. Default is `300` (5 minutes).
- `MARKETS_TO_ANALYZE`: The number of recent markets to send to the AI for analysis. Default is `25`.
- `USE_SWARM_MODE`: A boolean to toggle the use of the AI swarm. Default is `True`.
- `TOP_MARKETS_COUNT`: The number of top consensus picks to identify and display. Default is `5`.

### AI Prompts

The AI prompts are a crucial part of the agent's predictive power and can be customized to fit your specific needs.

- `MARKET_ANALYSIS_SYSTEM_PROMPT`: The system prompt for the individual AI models in the swarm.
- `CONSENSUS_AI_PROMPT_TEMPLATE`: The prompt for the consensus AI that synthesizes the responses from the swarm.

By modifying these prompts, you can:

- Adjust the analysis criteria (e.g., technical, fundamental, sentiment).
- Incorporate market-specific knowledge.
- Change the agent's risk tolerance.
- Focus on particular types of markets.

### Category Filters

The agent can be configured to ignore certain market categories by adding keywords to the following lists:

- `IGNORE_CRYPTO_KEYWORDS`: A list of keywords to filter out crypto-related markets.
- `IGNORE_SPORTS_KEYWORDS`: A list of keywords to filter out sports-related markets.

## How It Works

The Polymarket Prediction Agent operates using three parallel threads:

1. **WebSocket Thread:** This thread is responsible for collecting trades from the Polymarket WebSocket in real-time and saving them to a CSV file.
2. **Status Thread:** This thread prints statistical updates to the console every 30 seconds, including the number of trades, filtered markets, and more.
3. **Analysis Thread:** This thread checks for new markets every 5 minutes and triggers the AI analysis when the number of new markets reaches the `NEW_MARKETS_FOR_ANALYSIS` threshold.

### AI Analysis Workflow

The AI analysis is performed as follows:

- **First Run:** The agent analyzes any existing markets upon its first run.
- **Subsequent Runs:** The agent waits for 25 new markets to be collected before performing another analysis.
- **Swarm Mode:** In swarm mode, the agent queries Claude, OpenAI, Groq, Gemini, DeepSeek, and XAI models in parallel.
- **Consensus:** After receiving responses from the swarm, a consensus mechanism identifies the top 5 markets with the strongest agreement among the models.

## Output Files

The agent generates three CSV files located in the `src/data/polymarket/` directory:

- `markets.csv`: Contains all the markets with trades over the configured minimum trade size.
- `predictions.csv`: Contains the predictions from all 25 markets for each analysis run.
- `consensus_picks.csv`: An append-only file that contains only the top 5 consensus picks from each analysis run.

## API Requirements

To use the Polymarket Prediction Agent, you need to add the following API keys to your `.env` file:

- `ANTHROPIC_KEY`
- `OPENAI_KEY`
- `GROQ_API_KEY`
- `GEMINI_KEY`
- `DEEPSEEK_KEY`
- `XAI_API_KEY`

If you are not using the swarm mode, you only need to provide the key for the specific AI model you are using.
