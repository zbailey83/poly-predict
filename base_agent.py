"""A base class for creating trading agents with unified exchange support.

This class provides a common structure for agents that interact with
cryptocurrency exchanges. It includes a basic initialization method, a way to
get active tokens, and a placeholder for the main agent logic.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from termcolor import cprint

class BaseAgent:
    """A base class for creating trading agents with unified exchange support.

    This class provides a common structure for agents that interact with
    cryptocurrency exchanges. It includes a basic initialization method, a way to
    get active tokens, and a placeholder for the main agent logic.
    """
    def __init__(self, agent_type, use_exchange_manager=False):
        """Initializes the BaseAgent.

        Args:
            agent_type (str): The type of agent being created (e.g., 'trading',
                'risk', 'strategy').
            use_exchange_manager (bool, optional): Whether to initialize the
                ExchangeManager for unified trading. Defaults to False.
        """
        self.type = agent_type
        self.start_time = datetime.now()
        self.em = None  # Exchange manager instance

        # Initialize exchange manager if requested
        if use_exchange_manager:
            try:
                from src.exchange_manager import ExchangeManager
                from src.config import EXCHANGE

                self.em = ExchangeManager()
                cprint(f"✅ {agent_type.capitalize()} agent initialized with {EXCHANGE} exchange", "green")

                # Store exchange type for convenience
                self.exchange = EXCHANGE

            except Exception as e:
                cprint(f"⚠️ Could not initialize ExchangeManager: {str(e)}", "yellow")
                cprint("   Falling back to direct nice_funcs imports", "yellow")

                # Fallback to direct imports
                from src import nice_funcs as n
                self.n = n
                self.exchange = 'solana'  # Default fallback

    def get_active_tokens(self):
        """Gets the list of active tokens based on the current exchange.

        This method attempts to dynamically import and use the `get_active_tokens`
        function from the configuration. If that fails, it falls back to a static
        list of monitored tokens.

        Returns:
            list: A list of token symbols.
        """
        try:
            from src.config import get_active_tokens
            return get_active_tokens()
        except:
            from src.config import MONITORED_TOKENS
            return MONITORED_TOKENS

    def run(self):
        """The main entry point for the agent's logic.

        This method should be implemented by subclasses to define the agent's
        primary behavior.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError("Each agent must implement its own run method")