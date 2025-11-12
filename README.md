# Poly-Predict: Polymarket Data Prediction and Trading AI Agent Swarm

Poly-Predict is a sophisticated Python-based framework for interacting with the Polymarket prediction market. It provides a swarm of AI agents that can monitor, analyze, and predict market outcomes in real-time. The framework is designed to be highly customizable and extensible, allowing you to create your own trading agents and strategies.

## Features

- **Real-time Market Monitoring:** The `PolymarketAgent` connects to the Polymarket WebSocket to receive live trade data.
- **AI-Powered Predictions:** The `SwarmAgent` can query multiple AI models in parallel to get diverse perspectives on market outcomes.
- **Tweet Generation:** The `TweetAgent` can generate tweets from a given text, allowing you to share your market insights with the world.
- **Comprehensive Data Logging:** All market data, AI predictions, and consensus picks are saved to CSV files for historical analysis.
- **Easy to Use:** The framework is designed to be easy to use, with a simple and intuitive API.
- **Highly Customizable:** The framework is highly customizable, allowing you to create your own trading agents and strategies.

## Getting Started

To get started with Poly-Predict, you'll need to have Python 3.7 or higher installed. You'll also need to have a Conda environment set up.

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/poly-predict.git
   ```

2. **Create a Conda Environment:**
   ```bash
   conda create --name poly-predict python=3.7
   conda activate poly-predict
   ```

3. **Install the Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Your API Keys:**
   You'll need to get API keys for the AI models you want to use. You can get these keys from the following websites:
   - [Anthropic](https://www.anthropic.com/)
   - [OpenAI](https://openai.com/)
   - [Groq](https://groq.com/)
   - [Gemini](https://gemini.google.com/)
   - [DeepSeek](https://www.deepseek.com/)
   - [XAI](https://x.ai/)

   Once you have your API keys, create a `.env` file in the root of the project and add the following lines:
   ```
   ANTHROPIC_KEY=your_key_here
   OPENAI_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   GEMINI_KEY=your_key_here
   DEEPSEEK_KEY=your_key_here
   XAI_API_KEY=your_key_here
   ```

5. **Run the Agents:**
   You can run the agents from the command line:
   ```bash
   python src/agents/polymarket_agent.py
   python src/agents/swarm_agent.py
   python src/agents/tweet_agent.py
   ```

## Agents

The Poly-Predict framework includes three agents:

- **`PolymarketAgent`:** This agent connects to the Polymarket WebSocket to receive live trade data. It filters trades based on size and other criteria, saves market data to a CSV file, and uses an AI model (or a swarm of models) to generate predictions on market outcomes.
- **`SwarmAgent`:** This agent can query multiple AI models in parallel to get diverse perspectives on a given prompt. It is an ideal solution for tasks that benefit from a variety of AI viewpoints, such as making trading decisions, validating strategies, or assessing risks.
- **`TweetAgent`:** This agent generates tweets from a given text input. It uses an AI model to create multiple tweets based on the content of the text, which can be provided as a string or read from a file.

## Contributing

We welcome contributions to the Poly-Predict project. If you have an idea for a new feature or have found a bug, please open an issue on our [GitHub repository](https://github.com/your-username/poly-predict/issues).

## License

Poly-Predict is licensed under the MIT License. See the `LICENSE` file for more information.
