"""
TradingGraph: Orchestrates the multi-agent trading system using LangChain and LangGraph.
Initializes LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
"""

import os
from typing import Dict

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from default_config import DEFAULT_CONFIG
from graph_setup import SetGraph
from graph_util import TechnicalTools


class TradingGraph:
    """
    Main orchestrator for the multi-agent trading system.
    Sets up LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
    """

    def __init__(self, config=None):
        # --- Configuration and LLMs ---
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        self.toolkit = TechnicalTools()

        # --- Create tool nodes for each agent ---
        self.tool_nodes = self._set_tool_nodes()

        # --- Graph logic and setup ---
        if self._get_api_key():
            self.refresh_llms()

    def _get_api_key(self):
        """
        Get API key from config or environment variable.

        Returns:
            str or None: The OpenAI API key if found, otherwise None.
        """
        # First check if API key is provided in config
        api_key = self.config.get("api_key")

        # If not in config, check environment variable
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        # Return the key if it's not a placeholder
        if api_key and api_key != "your-openai-api-key-here" and api_key != "":
            return api_key

        return None

    def _set_tool_nodes(self) -> Dict[str, ToolNode]:
        """
        Define tool nodes for each agent type (indicator, pattern, trend).
        """
        return {
            "indicator": ToolNode(
                [
                    self.toolkit.compute_macd,
                    self.toolkit.compute_roc,
                    self.toolkit.compute_rsi,
                    self.toolkit.compute_stoch,
                    self.toolkit.compute_willr,
                ]
            ),
            "pattern": ToolNode(
                [
                    self.toolkit.generate_kline_image,
                ]
            ),
            "trend": ToolNode([self.toolkit.generate_trend_image]),
        }

    def _get_base_url(self):
        """Get API base_url."""
        return self.config.get("base_url") or os.environ.get("OPENAI_BASE_URL")

    def refresh_llms(self):
        """
        Refresh the LLM objects with the current API key from environment.
        This is called when the API key is updated.
        """
        # Get the current API key with validation
        api_key = self._get_api_key()
        base_url = self._get_base_url()

        llm_params = {"api_key": api_key}
        if base_url:
            llm_params["base_url"] = base_url

        # Recreate LLM objects with explicit API key and config values
        self.agent_llm = ChatOpenAI(
            model=self.config.get("agent_llm_model", "gpt-4o-mini"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
            **llm_params,
        )
        self.graph_llm = ChatOpenAI(
            model=self.config.get("graph_llm_model", "gpt-4o"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
            **llm_params,
        )

        # Recreate the graph setup with new LLMs
        self.graph_setup = SetGraph(
            self.agent_llm,
            self.graph_llm,
            self.toolkit,
            self.tool_nodes,
        )

        # Recreate the main graph
        self.graph = self.graph_setup.set_graph()

    def update_llm_settings(self, api_key: str, base_url: str = None, agent_llm_model: str = None, graph_llm_model: str = None):
        """
        Update LLM settings and refresh the clients.

        Args:
            api_key (str): The new OpenAI API key.
            base_url (str, optional): The new API base_url.
            agent_llm_model (str, optional): The new model name for the agent LLM.
            graph_llm_model (str, optional): The new model name for the graph LLM.
        """
        if api_key:
            self.config["api_key"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key

        if base_url:
            self.config["base_url"] = base_url
            os.environ["OPENAI_BASE_URL"] = base_url
        else:
            self.config["base_url"] = None
            if "OPENAI_BASE_URL" in os.environ:
                del os.environ["OPENAI_BASE_URL"]

        if agent_llm_model:
            self.config["agent_llm_model"] = agent_llm_model

        if graph_llm_model:
            self.config["graph_llm_model"] = graph_llm_model

        self.refresh_llms()
