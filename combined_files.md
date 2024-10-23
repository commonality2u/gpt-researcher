

# D:\Documents\Github\gpt-researcher-wes\gpt_researcher\retrievers\__init__.py

from .arxiv.arxiv import ArxivSearch
from .bing.bing import BingSearch
from .custom.custom import CustomRetriever
from .duckduckgo.duckduckgo import Duckduckgo
from .google.google import GoogleSearch
from .pubmed_central.pubmed_central import PubMedCentralSearch
from .searx.searx import SearxSearch
from .semantic_scholar.semantic_scholar import SemanticScholarSearch
from .searchapi.searchapi import SearchApiSearch
from .serpapi.serpapi import SerpApiSearch
from .serper.serper import SerperSearch
from .tavily.tavily_search import TavilySearch
from .exa.exa import ExaSearch

__all__ = [
    "TavilySearch",
    "CustomRetriever",
    "Duckduckgo",
    "SearchApiSearch",
    "SerperSearch",
    "SerpApiSearch",
    "GoogleSearch",
    "SearxSearch",
    "BingSearch",
    "ArxivSearch",
    "SemanticScholarSearch",
    "PubMedCentralSearch",
    "ExaSearch"
]


# D:\Documents\Github\gpt-researcher-wes\gpt_researcher\master\agent\master.py

from typing import Optional, List, Dict, Any, Set

from ...config import Config
from ...memory import Memory
from ...utils.enum import ReportSource, ReportType, Tone
from ...llm_provider import GenericLLMProvider
from ..agent.researcher import ResearchConductor
from ..agent.scraper import ReportScraper
from ..agent.writer import ReportGenerator
from ..agent.context_manager import ContextManager
from ..actions import get_retrievers, choose_agent
from ...vector_store import VectorStoreWrapper


class GPTResearcher:
    def __init__(
        self,
        query: str,
        report_type: str = ReportType.ResearchReport.value,
        report_format: str = "markdown",  # Add this line
        report_source: str = ReportSource.Web.value,
        tone: Tone = Tone.Objective,
        source_urls=None,
        documents=None,
        vector_store=None,
        vector_store_filter=None,
        config_path=None,
        websocket=None,
        agent=None,
        role=None,
        parent_query: str = "",
        subtopics: list = [],
        visited_urls: set = set(),
        verbose: bool = True,
        context=[],
        headers: dict = None,
        max_subtopics: int = 5,  # Add this line
    ):
        self.query = query
        self.report_type = report_type
        self.cfg = Config(config_path)
        self.llm = GenericLLMProvider(self.cfg)
        self.report_source = getattr(
            self.cfg, 'report_source', None) or report_source
        self.report_format = report_format
        self.max_subtopics = max_subtopics
        self.tone = tone if isinstance(tone, Tone) else Tone.Objective
        self.source_urls = source_urls
        self.documents = documents
        self.vector_store = VectorStoreWrapper(vector_store) if vector_store else None
        self.vector_store_filter = vector_store_filter
        self.websocket = websocket
        self.agent = agent
        self.role = role
        self.parent_query = parent_query
        self.subtopics = subtopics
        self.visited_urls = visited_urls
        self.verbose = verbose
        self.context = context
        self.headers = headers or {}
        self.research_costs = 0.0
        self.retrievers = get_retrievers(self.headers, self.cfg)
        self.memory = Memory(
            getattr(self.cfg, 'embedding_provider', None), self.headers)

        # Initialize components
        self.research_conductor = ResearchConductor(self)
        self.report_generator = ReportGenerator(self)
        self.scraper = ReportScraper(self)
        self.context_manager = ContextManager(self)

    async def conduct_research(self):
        if not (self.agent and self.role):
            self.agent, self.role = await choose_agent(
                query=self.query,
                cfg=self.cfg,
                parent_query=self.parent_query,
                cost_callback=self.add_costs,
                headers=self.headers,
            )

        self.context = await self.research_conductor.conduct_research()
        return self.context

    async def write_report(self, existing_headers: list = [], relevant_written_contents: list = [], ext_context=None) -> str:
        return await self.report_generator.write_report(
            existing_headers,
            relevant_written_contents,
            ext_context or self.context
        )

    async def write_report_conclusion(self, report_body: str) -> str:
        return await self.report_generator.write_report_conclusion(report_body)

    async def write_introduction(self):
        return await self.report_generator.write_introduction()

    async def get_subtopics(self):
        return await self.report_generator.get_subtopics()

    async def get_draft_section_titles(self, current_subtopic: str):
        return await self.report_generator.get_draft_section_titles(current_subtopic)

    async def get_similar_written_contents_by_draft_section_titles(
        self,
        current_subtopic: str,
        draft_section_titles: List[str],
        written_contents: List[Dict],
        max_results: int = 10
    ) -> List[str]:
        return await self.context_manager.get_similar_written_contents_by_draft_section_titles(
            current_subtopic,
            draft_section_titles,
            written_contents,
            max_results
        )

    # Utility methods
    def get_source_urls(self) -> list:
        return list(self.visited_urls)

    def get_research_context(self) -> list:
        return self.context

    def get_costs(self) -> float:
        return self.research_costs

    def set_verbose(self, verbose: bool):
        self.verbose = verbose

    def add_costs(self, cost: float) -> None:
        if not isinstance(cost, (float, int)):
            raise ValueError("Cost must be an integer or float")
        self.research_costs += cost


# D:\Documents\Github\gpt-researcher-wes\gpt_researcher\master\actions\__init__.py

from .retriever import get_retriever, get_retrievers
from .query_processing import get_sub_queries, extract_json_with_regex, choose_agent
from .web_scraping import scrape_urls
from .report_generation import write_conclusion, summarize_url, generate_draft_section_titles, generate_report, write_report_introduction
from .markdown_processing import extract_headers, extract_sections, table_of_contents, add_references
from .utils import stream_output

__all__ = [
    "get_retriever",
    "get_retrievers",
    "get_sub_queries",
    "extract_json_with_regex",
    "scrape_urls",
    "write_conclusion",
    "summarize_url",
    "generate_draft_section_titles",
    "generate_report",
    "write_report_introduction",
    "extract_headers",
    "extract_sections",
    "table_of_contents",
    "add_references",
    "stream_output",
    "choose_agent"
]

# D:\Documents\Github\gpt-researcher-wes\gpt_researcher\master\actions\query_processing.py

import json
import re
import json_repair
from ...utils.llm import create_chat_completion
from ..prompts import auto_agent_instructions, generate_search_queries_prompt


async def choose_agent(
    query, cfg, parent_query=None, cost_callback: callable = None, headers=None
):
    """
    Chooses the agent automatically
    Args:
        parent_query: In some cases the research is conducted on a subtopic from the main query.
        The parent query allows the agent to know the main context for better reasoning.
        query: original query
        cfg: Config
        cost_callback: callback for calculating llm costs

    Returns:
        agent: Agent name
        agent_role_prompt: Agent role prompt
    """
    query = f"{parent_query} - {query}" if parent_query else f"{query}"
    response = None  # Initialize response to ensure it's defined

    try:
        response = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{auto_agent_instructions()}"},
                {"role": "user", "content": f"task: {query}"},
            ],
            temperature=0.15,
            llm_provider=cfg.smart_llm_provider,
            llm_kwargs=cfg.llm_kwargs,
            cost_callback=cost_callback,
        )

        agent_dict = json.loads(response)
        return agent_dict["server"], agent_dict["agent_role_prompt"]

    except Exception as e:
        print("⚠️ Error in reading JSON, attempting to repair JSON")
        return await handle_json_error(response)


async def handle_json_error(response):
    try:
        agent_dict = json_repair.loads(response)
        if agent_dict.get("server") and agent_dict.get("agent_role_prompt"):
            return agent_dict["server"], agent_dict["agent_role_prompt"]
    except Exception as e:
        print(f"Error using json_repair: {e}")

    json_string = extract_json_with_regex(response)
    if json_string:
        try:
            json_data = json.loads(json_string)
            return json_data["server"], json_data["agent_role_prompt"]
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    print("No JSON found in the string. Falling back to Default Agent.")
    return "Default Agent", (
        "You are an AI critical thinker research assistant. Your sole purpose is to write well written, "
        "critically acclaimed, objective and structured reports on given text."
    )


def extract_json_with_regex(response):
    json_match = re.search(r"{.*?}", response, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return None


async def get_sub_queries(
    query: str,
    agent_role_prompt: str,
    cfg,
    parent_query: str,
    report_type: str,
    cost_callback: callable = None,
):
    """
    Gets the sub queries
    Args:
        query: original query
        agent_role_prompt: agent role prompt
        cfg: Config
        parent_query:
        report_type:
        cost_callback:

    Returns:
        sub_queries: List of sub queries

    """
    max_research_iterations = cfg.max_iterations if cfg.max_iterations else 1
    response = await create_chat_completion(
        model=cfg.smart_llm_model,
        messages=[
            {"role": "system", "content": f"{agent_role_prompt}"},
            {
                "role": "user",
                "content": generate_search_queries_prompt(
                    query,
                    parent_query,
                    report_type,
                    max_iterations=max_research_iterations,
                ),
            },
        ],
        temperature=0.1,
        llm_provider=cfg.smart_llm_provider,
        llm_kwargs=cfg.llm_kwargs,
        cost_callback=cost_callback,
    )

    sub_queries = json_repair.loads(response)

    return sub_queries


# D:\Documents\Github\gpt-researcher-wes\gpt_researcher\master\actions\report_generation.py

import asyncio
from typing import List, Dict, Any
from ...config.config import Config
from ...utils.llm import create_chat_completion
from ...utils.logger import get_formatted_logger
from ..prompts import (
    generate_report_introduction,
    generate_draft_titles_prompt,
    generate_report_conclusion,
    get_prompt_by_report_type,
)
from ...utils.enum import Tone

logger = get_formatted_logger()


async def write_report_introduction(
    query: str,
    context: str,
    agent_role_prompt: str,
    config: Config,
    websocket=None,
    cost_callback: callable = None
) -> str:
    """
    Generate an introduction for the report.

    Args:
        query (str): The research query.
        context (str): Context for the report.
        role (str): The role of the agent.
        config (Config): Configuration object.
        websocket: WebSocket connection for streaming output.
        cost_callback (callable, optional): Callback for calculating LLM costs.

    Returns:
        str: The generated introduction.
    """
    try:
        introduction = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {"role": "user", "content": generate_report_introduction(
                    query, context)},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
        )
        return introduction
    except Exception as e:
        logger.error(f"Error in generating report introduction: {e}")
    return ""


async def write_conclusion(
    query: str,
    context: str,
    agent_role_prompt: str,
    config: Config,
    websocket=None,
    cost_callback: callable = None
) -> str:
    """
    Write a conclusion for the report.

    Args:
        query (str): The research query.
        context (str): Context for the report.
        role (str): The role of the agent.
        config (Config): Configuration object.
        websocket: WebSocket connection for streaming output.
        cost_callback (callable, optional): Callback for calculating LLM costs.

    Returns:
        str: The generated conclusion.
    """
    try:
        conclusion = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {"role": "user", "content": generate_report_conclusion(query, context)},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
        )
        return conclusion
    except Exception as e:
        logger.error(f"Error in writing conclusion: {e}")
    return ""


async def summarize_url(
    url: str,
    content: str,
    role: str,
    config: Config,
    websocket=None,
    cost_callback: callable = None
) -> str:
    """
    Summarize the content of a URL.

    Args:
        url (str): The URL to summarize.
        content (str): The content of the URL.
        role (str): The role of the agent.
        config (Config): Configuration object.
        websocket: WebSocket connection for streaming output.
        cost_callback (callable, optional): Callback for calculating LLM costs.

    Returns:
        str: The summarized content.
    """
    try:
        summary = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": f"Summarize the following content from {url}:\n\n{content}"},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
        )
        return summary
    except Exception as e:
        logger.error(f"Error in summarizing URL: {e}")
    return ""


async def generate_draft_section_titles(
    query: str,
    current_subtopic: str,
    context: str,
    role: str,
    config: Config,
    websocket=None,
    cost_callback: callable = None
) -> List[str]:
    """
    Generate draft section titles for the report.

    Args:
        query (str): The research query.
        context (str): Context for the report.
        role (str): The role of the agent.
        config (Config): Configuration object.
        websocket: WebSocket connection for streaming output.
        cost_callback (callable, optional): Callback for calculating LLM costs.

    Returns:
        List[str]: A list of generated section titles.
    """
    try:
        section_titles = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": generate_draft_titles_prompt(
                    current_subtopic, query, context)},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            stream=True,
            websocket=None,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
        )
        return section_titles.split("\n")
    except Exception as e:
        logger.error(f"Error in generating draft section titles: {e}")
    return []


async def generate_report(
    query: str,
    context,
    agent_role_prompt: str,
    report_type: str,
    tone: Tone,
    report_source: str,
    websocket,
    cfg,
    main_topic: str = "",
    existing_headers: list = [],
    relevant_written_contents: list = [],
    cost_callback: callable = None,
    headers=None,
):
    """
    generates the final report
    Args:
        query:
        context:
        agent_role_prompt:
        report_type:
        websocket:
        tone:
        cfg:
        main_topic:
        existing_headers:
        relevant_written_contents:
        cost_callback:

    Returns:
        report:

    """
    generate_prompt = get_prompt_by_report_type(report_type)
    report = ""

    if report_type == "subtopic_report":
        content = f"{generate_prompt(query, existing_headers, relevant_written_contents, main_topic, context, report_format=cfg.report_format, tone=tone, total_words=cfg.total_words)}"
    else:
        content = f"{generate_prompt(query, context, report_source, report_format=cfg.report_format, tone=tone, total_words=cfg.total_words)}"
    try:
        report = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {"role": "user", "content": content},
            ],
            temperature=0.35,
            llm_provider=cfg.smart_llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=cfg.smart_token_limit,
            llm_kwargs=cfg.llm_kwargs,
            cost_callback=cost_callback,
        )
    except Exception as e:
        print(f"Error in generate_report: {e}")

    return report


# D:\Documents\Github\gpt-researcher-wes\gpt_researcher\master\agent\writer.py

from typing import Dict, Optional

from ...utils.llm import construct_subtopics
from ..actions import (
    stream_output,
    generate_report,
    generate_draft_section_titles,
    write_report_introduction,
    write_conclusion
)


class ReportGenerator:
    """Generates reports based on research data."""

    def __init__(self, researcher):
        self.researcher = researcher
        self.research_params = {
            "query": self.researcher.query,
            "agent_role_prompt": self.researcher.cfg.agent_role or self.researcher.role,
            "report_type": self.researcher.report_type,
            "report_source": self.researcher.report_source,
            "tone": self.researcher.tone,
            "websocket": self.researcher.websocket,
            "cfg": self.researcher.cfg,
            "headers": self.researcher.headers,
        }

    async def write_report(self, existing_headers: list = [], relevant_written_contents: list = [], ext_context=None) -> str:
        """
        Write a report based on existing headers and relevant contents.

        Args:
            existing_headers (list): List of existing headers.
            relevant_written_contents (list): List of relevant written contents.
            ext_context (Optional): External context, if any.

        Returns:
            str: The generated report.
        """
        context = ext_context or self.researcher.context
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_report",
                f"✍️ Writing report for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        report_params = self.research_params.copy()
        report_params["context"] = context

        if self.researcher.report_type == "subtopic_report":
            report_params.update({
                "main_topic": self.researcher.parent_query,
                "existing_headers": existing_headers,
                "relevant_written_contents": relevant_written_contents,
                "cost_callback": self.researcher.add_costs,
            })
        else:
            report_params["cost_callback"] = self.researcher.add_costs

        report = await generate_report(**report_params)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "report_written",
                f"📝 Report written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return report

    async def write_report_conclusion(self, report_content: str) -> str:
        """
        Write the conclusion for the report.

        Args:
            report_content (str): The content of the report.

        Returns:
            str: The generated conclusion.
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_conclusion",
                f"✍️ Writing conclusion for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        conclusion = await write_conclusion(
            query=self.researcher.query,
            context=report_content,
            config=self.researcher.cfg,
            agent_role_prompt=self.researcher.cfg.agent_role or self.researcher.role,
            cost_callback=self.researcher.add_costs,
            websocket=self.researcher.websocket,
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "conclusion_written",
                f"📝 Conclusion written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return conclusion

    async def write_introduction(self):
        """Write the introduction section of the report."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "writing_introduction",
                f"✍️ Writing introduction for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        introduction = await write_report_introduction(
            query=self.researcher.query,
            context=self.researcher.context,
            agent_role_prompt=self.researcher.cfg.agent_role or self.researcher.role,
            config=self.researcher.cfg,
            websocket=self.researcher.websocket,
            cost_callback=self.researcher.add_costs,
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "introduction_written",
                f"📝 Introduction written for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return introduction

    async def get_subtopics(self):
        """Retrieve subtopics for the research."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "generating_subtopics",
                f"🌳 Generating subtopics for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        subtopics = await construct_subtopics(
            task=self.researcher.query,
            data=self.researcher.context,
            config=self.researcher.cfg,
            subtopics=self.researcher.subtopics,
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subtopics_generated",
                f"📊 Subtopics generated for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return subtopics

    async def get_draft_section_titles(self, current_subtopic: str):
        """Generate draft section titles for the report."""
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "generating_draft_sections",
                f"📑 Generating draft section titles for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        draft_section_titles = await generate_draft_section_titles(
            query=self.researcher.query,
            current_subtopic=current_subtopic,
            context=self.researcher.context,
            role=self.researcher.cfg.agent_role or self.researcher.role,
            websocket=self.researcher.websocket,
            config=self.researcher.cfg,
            cost_callback=self.researcher.add_costs,
        )

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "draft_sections_generated",
                f"🗂️ Draft section titles generated for '{self.researcher.query}'",
                self.researcher.websocket,
            )

        return draft_section_titles


# D:\Documents\Github\gpt-researcher-wes\gpt_researcher\master\prompts.py

import warnings
from datetime import date, datetime, timezone

from ..utils.enum import ReportSource, ReportType, Tone


def generate_search_queries_prompt(
    question: str,
    parent_query: str,
    report_type: str,
    max_iterations: int = 3,
):
    """Generates the search queries prompt for the given question.
    Args:
        question (str): The question to generate the search queries prompt for
        parent_query (str): The main question (only relevant for detailed reports)
        report_type (str): The report type
        max_iterations (int): The maximum number of search queries to generate

    Returns: str: The search queries prompt for the given question
    """

    if (
        report_type == ReportType.DetailedReport.value
        or report_type == ReportType.SubtopicReport.value
    ):
        task = f"{parent_query} - {question}"
    else:
        task = question

    return (
        f'Write {max_iterations} google search queries to search online that form an objective opinion from the following task: "{task}"\n'
        f"Assume the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.\n"
        f'You must respond with a list of strings in the following format: ["query 1", "query 2", "query 3"].\n'
        f"The response should contain ONLY the list."
    )


def generate_report_prompt(
    question: str,
    context,
    report_source: str,
    report_format="apa",
    total_words=1000,
    tone=None,
):
    """Generates the report prompt for the given question and research summary.
    Args: question (str): The question to generate the report prompt for
            research_summary (str): The research summary to generate the report prompt for
    Returns: str: The report prompt for the given question and research summary
    """

    reference_prompt = ""
    if report_source == ReportSource.Web.value:
        reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report: 

eg: Author, A. A. (Year, Month Date). Title of web page. Website Name. [url website](url)
"""
    else:
        reference_prompt = f"""
You MUST write all used source document names at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each."
"""

    tone_prompt = f"Write the report in a {tone.value} tone." if tone else ""

    return f"""
Information: "{context}"
---
Using the above information, answer the following query or task: "{question}" in a detailed report --
The report should focus on the answer to the query, should be well structured, informative, 
in-depth, and comprehensive, with facts and numbers if available and at least {total_words} words.
You should strive to write the report as long as you can using all relevant and necessary information provided.

Please follow all of the following guidelines in your report:
- You MUST determine your own concrete and valid opinion based on the given information. Do NOT defer to general and meaningless conclusions.
- You MUST write the report with markdown syntax and {report_format} format.
- You MUST prioritize the relevance, reliability, and significance of the sources you use. Choose trusted sources over less reliable ones.
- You must also prioritize new articles over older articles if the source can be trusted.
- Use in-text citation references in {report_format} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url)).
- Don't forget to add a reference list at the end of the report in {report_format} format and full url links without hyperlinks.
- {reference_prompt}
- {tone_prompt}

Please do your best, this is very important to my career.
Assume that the current date is {date.today()}.
"""


def generate_resource_report_prompt(
    question, context, report_source: str, report_format="apa", tone=None, total_words=1000
):
    """Generates the resource report prompt for the given question and research summary.

    Args:
        question (str): The question to generate the resource report prompt for.
        context (str): The research summary to generate the resource report prompt for.

    Returns:
        str: The resource report prompt for the given question and research summary.
    """

    reference_prompt = ""
    if report_source == ReportSource.Web.value:
        reference_prompt = f"""
            You MUST include all relevant source urls.
            Every url should be hyperlinked: [url website](url)
            """
    else:
        reference_prompt = f"""
            You MUST write all used source document names at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each."
        """

    return (
        f'"""{context}"""\n\nBased on the above information, generate a bibliography recommendation report for the following'
        f' question or topic: "{question}". The report should provide a detailed analysis of each recommended resource,'
        " explaining how each source can contribute to finding answers to the research question.\n"
        "Focus on the relevance, reliability, and significance of each source.\n"
        "Ensure that the report is well-structured, informative, in-depth, and follows Markdown syntax.\n"
        "Include relevant facts, figures, and numbers whenever available.\n"
        f"The report should have a minimum length of {total_words} words.\n"
        "You MUST include all relevant source urls."
        "Every url should be hyperlinked: [url website](url)"
        f"{reference_prompt}"
    )


def generate_custom_report_prompt(
    query_prompt, context, report_source: str, report_format="apa", tone=None, total_words=1000
):
    return f'"{context}"\n\n{query_prompt}'


def generate_outline_report_prompt(
    question, context, report_source: str, report_format="apa", tone=None,  total_words=1000
):
    """Generates the outline report prompt for the given question and research summary.
    Args: question (str): The question to generate the outline report prompt for
            research_summary (str): The research summary to generate the outline report prompt for
    Returns: str: The outline report prompt for the given question and research summary
    """

    return (
        f'"""{context}""" Using the above information, generate an outline for a research report in Markdown syntax'
        f' for the following question or topic: "{question}". The outline should provide a well-structured framework'
        " for the research report, including the main sections, subsections, and key points to be covered."
        f" The research report should be detailed, informative, in-depth, and a minimum of {total_words} words."
        " Use appropriate Markdown syntax to format the outline and ensure readability."
    )


def get_report_by_type(report_type: str):
    report_type_mapping = {
        ReportType.ResearchReport.value: generate_report_prompt,
        ReportType.ResourceReport.value: generate_resource_report_prompt,
        ReportType.OutlineReport.value: generate_outline_report_prompt,
        ReportType.CustomReport.value: generate_custom_report_prompt,
        ReportType.SubtopicReport.value: generate_subtopic_report_prompt,
    }
    return report_type_mapping[report_type]


def auto_agent_instructions():
    return """
This task involves researching a given topic, regardless of its complexity or the availability of a definitive answer. The research is conducted by a specific server, defined by its type and role, with each server requiring distinct instructions.
Agent
The server is determined by the field of the topic and the specific name of the server that could be utilized to research the topic provided. Agents are categorized by their area of expertise, and each server type is associated with a corresponding emoji.

examples:
task: "should I invest in apple stocks?"
response: 
{
    "server": "💰 Finance Agent",
    "agent_role_prompt: "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends."
}
task: "could reselling sneakers become profitable?"
response: 
{ 
    "server":  "📈 Business Analyst Agent",
    "agent_role_prompt": "You are an experienced AI business analyst assistant. Your main objective is to produce comprehensive, insightful, impartial, and systematically structured business reports based on provided business data, market trends, and strategic analysis."
}
task: "what are the most interesting sites in Tel Aviv?"
response:
{
    "server:  "🌍 Travel Agent",
    "agent_role_prompt": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights."
}
"""


def generate_summary_prompt(query, data):
    """Generates the summary prompt for the given question and text.
    Args: question (str): The question to generate the summary prompt for
            text (str): The text to generate the summary prompt for
    Returns: str: The summary prompt for the given question and text
    """

    return (
        f'{data}\n Using the above text, summarize it based on the following task or query: "{query}".\n If the '
        f"query cannot be answered using the text, YOU MUST summarize the text in short.\n Include all factual "
        f"information such as numbers, stats, quotes, etc if available. "
    )


################################################################################################

# DETAILED REPORT PROMPTS


def generate_subtopics_prompt() -> str:
    return """
Provided the main topic:

{task}

and research data:

{data}

- Construct a list of subtopics which indicate the headers of a report document to be generated on the task. 
- These are a possible list of subtopics : {subtopics}.
- There should NOT be any duplicate subtopics.
- Limit the number of subtopics to a maximum of {max_subtopics}
- Finally order the subtopics by their tasks, in a relevant and meaningful order which is presentable in a detailed report

"IMPORTANT!":
- Every subtopic MUST be relevant to the main topic and provided research data ONLY!

{format_instructions}
"""


def generate_subtopic_report_prompt(
    current_subtopic,
    existing_headers: list,
    relevant_written_contents: list,
    main_topic: str,
    context,
    report_format: str = "apa",
    max_subsections=5,
    total_words=800,
    tone: Tone = Tone.Objective,
) -> str:
    return f"""
Context:
"{context}"

Main Topic and Subtopic:
Using the latest information available, construct a detailed report on the subtopic: {current_subtopic} under the main topic: {main_topic}.
You must limit the number of subsections to a maximum of {max_subsections}.

Content Focus:
- The report should focus on answering the question, be well-structured, informative, in-depth, and include facts and numbers if available.
- Use markdown syntax and follow the {report_format.upper()} format.

IMPORTANT:Content and Sections Uniqueness:
- This part of the instructions is crucial to ensure the content is unique and does not overlap with existing reports.
- Carefully review the existing headers and existing written contents provided below before writing any new subsections.
- Prevent any content that is already covered in the existing written contents.
- Do not use any of the existing headers as the new subsection headers.
- Do not repeat any information already covered in the existing written contents or closely related variations to avoid duplicates.
- If you have nested subsections, ensure they are unique and not covered in the existing written contents.
- Ensure that your content is entirely new and does not overlap with any information already covered in the previous subtopic reports.

"Existing Subtopic Reports":
- Existing subtopic reports and their section headers:

    {existing_headers}

- Existing written contents from previous subtopic reports:

    {relevant_written_contents}

"Structure and Formatting":
- As this sub-report will be part of a larger report, include only the main body divided into suitable subtopics without any introduction or conclusion section.

- You MUST include markdown hyperlinks to relevant source URLs wherever referenced in the report, for example:

    ### Section Header
    
    This is a sample text. ([url website](url))

- Use H2 for the main subtopic header (##) and H3 for subsections (###).
- Use smaller Markdown headers (e.g., H2 or H3) for content structure, avoiding the largest header (H1) as it will be used for the larger report's heading.
- Organize your content into distinct sections that complement but do not overlap with existing reports.
- When adding similar or identical subsections to your report, you should clearly indicate the differences between and the new content and the existing written content from previous subtopic reports. For example:

    ### New header (similar to existing header)

    While the previous section discussed [topic A], this section will explore [topic B]."

"Date":
Assume the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.

"IMPORTANT!":
- The focus MUST be on the main topic! You MUST Leave out any information un-related to it!
- Must NOT have any introduction, conclusion, summary or reference section.
- You MUST include hyperlinks with markdown syntax ([url website](url)) related to the sentences wherever necessary.
- You MUST mention the difference between the existing content and the new content in the report if you are adding the similar or same subsections wherever necessary.
- The report should have a minimum length of {total_words} words.
- Use an {tone.value} tone throughout the report.

Do NOT add a conclusion section.
"""


def generate_draft_titles_prompt(
    current_subtopic: str,
    main_topic: str,
    context: str,
    max_subsections: int = 5
) -> str:
    return f"""
"Context":
"{context}"

"Main Topic and Subtopic":
Using the latest information available, construct a draft section title headers for a detailed report on the subtopic: {current_subtopic} under the main topic: {main_topic}.

"Task":
1. Create a list of draft section title headers for the subtopic report.
2. Each header should be concise and relevant to the subtopic.
3. The header should't be too high level, but detailed enough to cover the main aspects of the subtopic.
4. Use markdown syntax for the headers, using H3 (###) as H1 and H2 will be used for the larger report's heading.
5. Ensure the headers cover main aspects of the subtopic.

"Structure and Formatting":
Provide the draft headers in a list format using markdown syntax, for example:

### Header 1
### Header 2
### Header 3

"IMPORTANT!":
- The focus MUST be on the main topic! You MUST Leave out any information un-related to it!
- Must NOT have any introduction, conclusion, summary or reference section.
- Focus solely on creating headers, not content.
"""


def generate_report_introduction(question: str, research_summary: str = "") -> str:
    return f"""{research_summary}\n 
Using the above latest information, Prepare a detailed report introduction on the topic -- {question}.
- The introduction should be succinct, well-structured, informative with markdown syntax.
- As this introduction will be part of a larger report, do NOT include any other sections, which are generally present in a report.
- The introduction should be preceded by an H1 heading with a suitable topic for the entire report.
- You must include hyperlinks with markdown syntax ([url website](url)) related to the sentences wherever necessary.
Assume that the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.
"""


def generate_report_conclusion(query: str, report_content: str) -> str:
    """
    Generate a concise conclusion summarizing the main findings and implications of a research report.

    Args:
        report_content (str): The content of the research report.

    Returns:
        str: A concise conclusion summarizing the report's main findings and implications.
    """
    prompt = f"""
    Based on the research report below and research task, please write a concise conclusion that summarizes the main findings and their implications:
    
    Research task: {query}
    
    Research Report: {report_content}

    Your conclusion should:
    1. Recap the main points of the research
    2. Highlight the most important findings
    3. Discuss any implications or next steps
    4. Be approximately 2-3 paragraphs long
    
    If there is no "## Conclusion" section title written at the end of the report, please add it to the top of your conclusion. 
    You must include hyperlinks with markdown syntax ([url website](url)) related to the sentences wherever necessary.
    
    Write the conclusion:
    """

    return prompt


report_type_mapping = {
    ReportType.ResearchReport.value: generate_report_prompt,
    ReportType.ResourceReport.value: generate_resource_report_prompt,
    ReportType.OutlineReport.value: generate_outline_report_prompt,
    ReportType.CustomReport.value: generate_custom_report_prompt,
    ReportType.SubtopicReport.value: generate_subtopic_report_prompt,
}


def get_prompt_by_report_type(report_type):
    prompt_by_type = report_type_mapping.get(report_type)
    default_report_type = ReportType.ResearchReport.value
    if not prompt_by_type:
        warnings.warn(
            f"Invalid report type: {report_type}.\n"
            f"Please use one of the following: {', '.join([enum_value for enum_value in report_type_mapping.keys()])}\n"
            f"Using default report type: {default_report_type} prompt.",
            UserWarning,
        )
        prompt_by_type = report_type_mapping.get(default_report_type)
    return prompt_by_type


# D:\Documents\Github\gpt-researcher-wes\backend\server\server.py

import json
import os
from typing import Dict, List

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, File, UploadFile, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend.server.server_utils import generate_report_files
from backend.server.websocket_manager import WebSocketManager
from multi_agents.main import run_research_task
from gpt_researcher.document.document import DocumentLoader
from gpt_researcher.master.actions import stream_output
from backend.server.server_utils import (
    sanitize_filename, handle_start_command, handle_human_feedback,
    generate_report_files, send_file_paths, get_config_dict,
    update_environment_variables, handle_file_upload, handle_file_deletion,
    execute_multi_agents, handle_websocket_communication, extract_command_data
)

# Models


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    agent: str


class ConfigRequest(BaseModel):
    ANTHROPIC_API_KEY: str
    TAVILY_API_KEY: str
    LANGCHAIN_TRACING_V2: str
    LANGCHAIN_API_KEY: str
    OPENAI_API_KEY: str
    DOC_PATH: str
    RETRIEVER: str
    GOOGLE_API_KEY: str = ''
    GOOGLE_CX_KEY: str = ''
    BING_API_KEY: str = ''
    SEARCHAPI_API_KEY: str = ''
    SERPAPI_API_KEY: str = ''
    SERPER_API_KEY: str = ''
    SEARX_URL: str = ''


# App initialization
app = FastAPI()

# Static files and templates
app.mount("/site", StaticFiles(directory="./frontend"), name="site")
app.mount("/static", StaticFiles(directory="./frontend/static"), name="static")
templates = Jinja2Templates(directory="./frontend")

# WebSocket manager
manager = WebSocketManager()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DOC_PATH = os.getenv("DOC_PATH", "./my-docs")

# Startup event


@app.on_event("startup")
def startup_event():
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    os.makedirs(DOC_PATH, exist_ok=True)

# Routes


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "report": None})


@app.get("/getConfig")
async def get_config(
    langchain_api_key: str = Header(None),
    openai_api_key: str = Header(None),
    tavily_api_key: str = Header(None),
    google_api_key: str = Header(None),
    google_cx_key: str = Header(None),
    bing_api_key: str = Header(None),
    searchapi_api_key: str = Header(None),
    serpapi_api_key: str = Header(None),
    serper_api_key: str = Header(None),
    searx_url: str = Header(None)
):
    return get_config_dict(
        langchain_api_key, openai_api_key, tavily_api_key,
        google_api_key, google_cx_key, bing_api_key,
        searchapi_api_key, serpapi_api_key, serper_api_key, searx_url
    )


@app.get("/files/")
async def list_files():
    files = os.listdir(DOC_PATH)
    print(f"Files in {DOC_PATH}: {files}")
    return {"files": files}


@app.post("/api/multi_agents")
async def run_multi_agents():
    return await execute_multi_agents(manager)


@app.post("/setConfig")
async def set_config(config: ConfigRequest):
    update_environment_variables(config.dict())
    return {"message": "Config updated successfully"}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return await handle_file_upload(file, DOC_PATH)


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    return await handle_file_deletion(filename, DOC_PATH)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await handle_websocket_communication(websocket, manager)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


# D:\Documents\Github\gpt-researcher-wes\main.py

from dotenv import load_dotenv

load_dotenv()

from backend.server.server import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# D:\Documents\Github\gpt-researcher-wes\requirements.txt

# dependencies
beautifulsoup4
colorama
md2pdf
python-dotenv
pyyaml
uvicorn
pydantic
fastapi
python-multipart
markdown
langchain>=0.2,<0.3
langchain_community>=0.2,<0.3
langchain-openai>=0.1,<0.2
langgraph
tiktoken
gpt-researcher
arxiv
PyMuPDF
requests
jinja2
aiofiles
mistune
python-docx
htmldocx
lxml_html_clean
websockets
unstructured
json_repair
json5
loguru

# uncomment for testing
# pytest
# pytest-asyncio


# D:\Documents\Github\gpt-researcher-wes\README.md

<div align="center">
<!--<h1 style="display: flex; align-items: center; gap: 10px;">
  <img src="https://github.com/assafelovic/gpt-researcher/assets/13554167/a45bac7c-092c-42e5-8eb6-69acbf20dde5" alt="Logo" width="25">
  GPT Researcher
</h1>-->
<img src="https://github.com/assafelovic/gpt-researcher/assets/13554167/20af8286-b386-44a5-9a83-3be1365139c3" alt="Logo" width="80">


####

[![Website](https://img.shields.io/badge/Official%20Website-gptr.dev-teal?style=for-the-badge&logo=world&logoColor=white&color=0891b2)](https://gptr.dev)
[![Documentation](https://img.shields.io/badge/Documentation-DOCS-f472b6?logo=googledocs&logoColor=white&style=for-the-badge)](https://docs.gptr.dev)
[![Discord Follow](https://dcbadge.vercel.app/api/server/QgZXvJAccX?style=for-the-badge&theme=clean-inverted&?compact=true)](https://discord.gg/QgZXvJAccX)

[![PyPI version](https://img.shields.io/pypi/v/gpt-researcher?logo=pypi&logoColor=white&style=flat)](https://badge.fury.io/py/gpt-researcher)
![GitHub Release](https://img.shields.io/github/v/release/assafelovic/gpt-researcher?style=flat&logo=github)
[![Open In Colab](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=grey&color=yellow&label=%20&style=flat&logoSize=40)](https://colab.research.google.com/github/assafelovic/gpt-researcher/blob/master/docs/docs/examples/pip-run.ipynb)
[![Docker Image Version](https://img.shields.io/docker/v/elestio/gpt-researcher/latest?arch=amd64&style=flat&logo=docker&logoColor=white&color=1D63ED)](https://hub.docker.com/r/gptresearcher/gpt-researcher)
[![Twitter Follow](https://img.shields.io/twitter/follow/assaf_elovic?style=social)](https://twitter.com/assaf_elovic)

[English](README.md) |
[中文](README-zh_CN.md) |
[日本語](README-ja_JP.md) |
[한국어](README-ko_KR.md)
</div>

# 🔎 GPT Researcher

**GPT Researcher is an autonomous agent designed for comprehensive web and local research on any given task.** 

The agent can produce detailed, factual and unbiased research reports, with customization options for focusing on relevant resources and outlines. Inspired by the recent [Plan-and-Solve](https://arxiv.org/abs/2305.04091) and [RAG](https://arxiv.org/abs/2005.11401) papers, GPT Researcher addresses issues of misinformation, speed, determinism and reliability, offering a more stable performance and increased speed through parallelized agent work, as opposed to synchronous operations.

**Our mission is to empower individuals and organizations with accurate, unbiased, and factual information by leveraging the power of AI.** 

## Why GPT Researcher?

- To form objective conclusions for manual research tasks can take time, sometimes weeks to find the right resources and information.
- Current LLMs are trained on past and outdated information, with heavy risks of hallucinations, making them almost irrelevant for research tasks.
- Current LLMs are limited to short token outputs which are not sufficient for long detailed research reports (2k+ words).
- Services that enable web search such as ChatGPT or Perplexity, only consider limited sources and content that in some cases result in misinformation and shallow results.
- Using only a selection of web sources can create bias in determining the right conclusions for research tasks.

## Demo
https://github.com/user-attachments/assets/092e9e71-7e27-475d-8c4f-9dddd28934a3

## Architecture
The main idea is to run "planner" and "execution" agents, whereas the planner generates questions to research, and the execution agents seek the most related information based on each generated research question. Finally, the planner filters and aggregates all related information and creates a research report. <br /> <br /> 
The agents leverage both `gpt-4o-mini` and `gpt-4o` (128K context) to complete a research task. We optimize for costs using each only when necessary. **The average research task takes around 2 minutes to complete, and costs ~$0.005.**

<div align="center">
<img align="center" height="600" src="https://github.com/assafelovic/gpt-researcher/assets/13554167/4ac896fd-63ab-4b77-9688-ff62aafcc527">
</div>



More specifically:
* Create a domain specific agent based on research query or task.
* Generate a set of research questions that together form an objective opinion on any given task. 
* For each research question, trigger a crawler agent that scrapes online resources for information relevant to the given task.
* For each scraped resources, summarize based on relevant information and keep track of its sources.
* Finally, filter and aggregate all summarized sources and generate a final research report.

## Tutorials
 - [How it Works](https://docs.gptr.dev/blog/building-gpt-researcher)
 - [How to Install](https://www.loom.com/share/04ebffb6ed2a4520a27c3e3addcdde20?sid=da1848e8-b1f1-42d1-93c3-5b0b9c3b24ea)
 - [Live Demo](https://www.loom.com/share/6a3385db4e8747a1913dd85a7834846f?sid=a740fd5b-2aa3-457e-8fb7-86976f59f9b8)

## Features
- 📝 Generate research, outlines, resources and lessons reports with local documents and web sources
- 📜 Can generate long and detailed research reports (over 2K words)
- 🌐 Aggregates over 20 web sources per research to form objective and factual conclusions
- 🖥️ Includes both lightweight (HTML/CSS/JS) and production ready (NextJS + Tailwind) UX/UI
- 🔍 Scrapes web sources with javascript support
- 📂 Keeps track and context and memory throughout the research process
- 📄 Export research reports to PDF, Word and more...

## 📖 Documentation

Please see [here](https://docs.gptr.dev/docs/gpt-researcher/getting-started/getting-started) for full documentation on:

- Getting started (installation, setting up the environment, simple examples)
- Customization and configuration
- How-To examples (demos, integrations, docker support)
- Reference (full API docs)

## ⚙️ Getting Started
### Installation
> **Step 0** - Install Python 3.11 or later. [See here](https://www.tutorialsteacher.com/python/install-python) for a step-by-step guide.

> **Step 1** - Download the project and navigate to its directory

```bash
git clone https://github.com/assafelovic/gpt-researcher.git
cd gpt-researcher
```

> **Step 3** - Set up API keys using two methods: exporting them directly or storing them in a `.env` file.

For Linux/Windows temporary setup, use the export method:

```bash
export OPENAI_API_KEY={Your OpenAI API Key here}
export TAVILY_API_KEY={Your Tavily API Key here}
```

For a more permanent setup, create a `.env` file in the current `gpt-researcher` directory and input the env vars (without `export`).

- The default LLM is [GPT](https://platform.openai.com/docs/guides/gpt), but you can use other LLMs such as `claude`, `ollama3`, `gemini`, `mistral` and more. To learn how to change the LLM provider, see the [LLMs documentation](https://docs.gptr.dev/docs/gpt-researcher/llms/llms) page. Please note: this project is optimized for OpenAI GPT models.
- The default retriever is [Tavily](https://app.tavily.com), but you can refer to other retrievers such as `duckduckgo`, `google`, `bing`, `searchapi`, `serper`, `searx`, `arxiv`, `exa` and more. To learn how to change the search provider, see the [retrievers documentation](https://docs.gptr.dev/docs/gpt-researcher/search-engines/retrievers) page.

### Quickstart

> **Step 1** - Install dependencies

```bash
pip install -r requirements.txt
```

> **Step 2** - Run the agent with FastAPI

```bash
python -m uvicorn main:app --reload
```

> **Step 3** - Go to http://localhost:8000 on any browser and enjoy researching!

<br />

**To learn how to get started with [Poetry](https://docs.gptr.dev/docs/gpt-researcher/getting-started/getting-started#poetry) or a [virtual environment](https://docs.gptr.dev/docs/gpt-researcher/getting-started/getting-started#virtual-environment) check out the [documentation](https://docs.gptr.dev/docs/gpt-researcher/getting-started/getting-started) page.**

### Run as PIP package
```bash
pip install gpt-researcher
```

```python
...
from gpt_researcher import GPTResearcher

query = "why is Nvidia stock going up?"
researcher = GPTResearcher(query=query, report_type="research_report")
# Conduct research on the given query
research_result = await researcher.conduct_research()
# Write the report
report = await researcher.write_report()
...
```

**For more examples and configurations, please refer to the [PIP documentation](https://docs.gptr.dev/docs/gpt-researcher/gptr/pip-package) page.**


## Run with Docker

> **Step 1** - [Install Docker](https://docs.gptr.dev/docs/gpt-researcher/getting-started/getting-started-with-docker)

> **Step 2** - Clone the '.env.example' file, add your API Keys to the cloned file and save the file as '.env'

> **Step 3** - Within the docker-compose file comment out services that you don't want to run with Docker.

```bash
docker-compose up --build
```

If that doesn't work, try running it without the dash:
```bash
docker compose up --build
```


> **Step 4** - By default, if you haven't uncommented anything in your docker-compose file, this flow will start 2 processes:
 - the Python server running on localhost:8000<br>
 - the React app running on localhost:3000<br>

Visit localhost:3000 on any browser and enjoy researching!



## 📄 Research on Local Documents

You can instruct the GPT Researcher to run research tasks based on your local documents. Currently supported file formats are: PDF, plain text, CSV, Excel, Markdown, PowerPoint, and Word documents.

Step 1: Add the env variable `DOC_PATH` pointing to the folder where your documents are located.

```bash
export DOC_PATH="./my-docs"
```

Step 2: 
 - If you're running the frontend app on localhost:8000, simply select "My Documents" from the the "Report Source" Dropdown Options.
 - If you're running GPT Researcher with the [PIP package](https://docs.tavily.com/docs/gpt-researcher/pip-package), pass the `report_source` argument as "local" when you instantiate the `GPTResearcher` class [code sample here](https://docs.gptr.dev/docs/gpt-researcher/context/tailored-research).


## 👪 Multi-Agent Assistant
As AI evolves from prompt engineering and RAG to multi-agent systems, we're excited to introduce our new multi-agent assistant built with [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/).

By using LangGraph, the research process can be significantly improved in depth and quality by leveraging multiple agents with specialized skills. Inspired by the recent [STORM](https://arxiv.org/abs/2402.14207) paper, this project showcases how a team of AI agents can work together to conduct research on a given topic, from planning to publication.

An average run generates a 5-6 page research report in multiple formats such as PDF, Docx and Markdown.

Check it out [here](https://github.com/assafelovic/gpt-researcher/tree/master/multi_agents) or head over to our [documentation](https://docs.gptr.dev/docs/gpt-researcher/multi_agents/langgraph) for more information.

## 🖥️ Frontend Applications

GPT-Researcher now features an enhanced frontend to improve the user experience and streamline the research process. The frontend offers:

- An intuitive interface for inputting research queries
- Real-time progress tracking of research tasks
- Interactive display of research findings
- Customizable settings for tailored research experiences

Two deployment options are available:
1. A lightweight static frontend served by FastAPI
2. A feature-rich NextJS application for advanced functionality

For detailed setup instructions and more information about the frontend features, please visit our [documentation page](https://docs.gptr.dev/docs/gpt-researcher/frontend/frontend).

## 🚀 Contributing
We highly welcome contributions! Please check out [contributing](https://github.com/assafelovic/gpt-researcher/blob/master/CONTRIBUTING.md) if you're interested.

Please check out our [roadmap](https://trello.com/b/3O7KBePw/gpt-researcher-roadmap) page and reach out to us via our [Discord community](https://discord.gg/QgZXvJAccX) if you're interested in joining our mission.
<a href="https://github.com/assafelovic/gpt-researcher/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=assafelovic/gpt-researcher" />
</a>
## ✉️ Support / Contact us
- [Community Discord](https://discord.gg/spBgZmm3Xe)
- Author Email: assaf.elovic@gmail.com

## 🛡 Disclaimer

This project, GPT Researcher, is an experimental application and is provided "as-is" without any warranty, express or implied. We are sharing codes for academic purposes under the Apache 2 license. Nothing herein is academic advice, and NOT a recommendation to use in academic or research papers.

Our view on unbiased research claims:
1. The main goal of GPT Researcher is to reduce incorrect and biased facts. How? We assume that the more sites we scrape the less chances of incorrect data. By scraping multiple sites per research, and choosing the most frequent information, the chances that they are all wrong is extremely low.
2. We do not aim to eliminate biases; we aim to reduce it as much as possible. **We are here as a community to figure out the most effective human/llm interactions.**
3. In research, people also tend towards biases as most have already opinions on the topics they research about. This tool scrapes many opinions and will evenly explain diverse views that a biased person would never have read.

---

<p align="center">
<a href="https://star-history.com/#assafelovic/gpt-researcher">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=assafelovic/gpt-researcher&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=assafelovic/gpt-researcher&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=assafelovic/gpt-researcher&type=Date" />
  </picture>
</a>
</p>


# D:\Documents\Github\gpt-researcher-wes\frontend\scripts.js

const GPTResearcher = (() => {
  const init = () => {
    // Not sure, but I think it would be better to add event handlers here instead of in the HTML
    //document.getElementById("startResearch").addEventListener("click", startResearch);
    document
      .getElementById('copyToClipboard')
      .addEventListener('click', copyToClipboard)

    updateState('initial')
  }

  const changeSource = () => {
    const report_source = document.querySelector('select[name="report_source"]').value
    if (report_source === 'sources') {
        document.getElementById('sources').style.display = 'block'
    } else {
        document.getElementById('sources').style.display = 'none'
    }
  }

  const startResearch = () => {
    document.getElementById('output').innerHTML = ''
    document.getElementById('reportContainer').innerHTML = ''
    updateState('in_progress')

    addAgentResponse({
      output: '🤔 Thinking about research questions for the task...',
    })

    listenToSockEvents()
  }

  const listenToSockEvents = () => {
    const { protocol, host, pathname } = window.location
    const ws_uri = `${
      protocol === 'https:' ? 'wss:' : 'ws:'
    }//${host}${pathname}ws`
    const converter = new showdown.Converter()
    const socket = new WebSocket(ws_uri)

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'logs') {
        addAgentResponse(data)
      } else if (data.type === 'report') {
        writeReport(data, converter)
      } else if (data.type === 'path') {
        updateState('finished')
        updateDownloadLink(data)
      }
    }

    socket.onopen = (event) => {
      const task = document.querySelector('input[name="task"]').value
      const report_type = document.querySelector(
        'select[name="report_type"]'
      ).value
      const report_source = document.querySelector(
        'select[name="report_source"]'
      ).value
      const tone = document.querySelector('select[name="tone"]').value
      const agent = document.querySelector('input[name="agent"]:checked').value
      let source_urls = tags

      if (report_source !== 'sources' && source_urls.length > 0) {
        source_urls = source_urls.slice(0, source_urls.length - 1)
      }

      const requestData = {
        task: task,
        report_type: report_type,
        report_source: report_source,
        source_urls: source_urls,
        tone: tone,
        agent: agent,
      }

      socket.send(`start ${JSON.stringify(requestData)}`)
    }
  }

  const addAgentResponse = (data) => {
    const output = document.getElementById('output')
    output.innerHTML += '<div class="agent_response">' + data.output + '</div>'
    output.scrollTop = output.scrollHeight
    output.style.display = 'block'
    updateScroll()
  }

  const writeReport = (data, converter) => {
    const reportContainer = document.getElementById('reportContainer')
    const markdownOutput = converter.makeHtml(data.output)
    reportContainer.innerHTML += markdownOutput
    updateScroll()
  }

  const updateDownloadLink = (data) => {
    const pdf_path = data.output.pdf
    const docx_path = data.output.docx
    const md_path = data.output.md;
    document.getElementById('downloadLink').setAttribute('href', pdf_path);
    document.getElementById('downloadLinkWord').setAttribute('href', docx_path);
    document.getElementById("downloadLinkMd").setAttribute("href", md_path);
  }

  const updateScroll = () => {
    window.scrollTo(0, document.body.scrollHeight)
  }

  const copyToClipboard = () => {
    const textarea = document.createElement('textarea')
    textarea.id = 'temp_element'
    textarea.style.height = 0
    document.body.appendChild(textarea)
    textarea.value = document.getElementById('reportContainer').innerText
    const selector = document.querySelector('#temp_element')
    selector.select()
    document.execCommand('copy')
    document.body.removeChild(textarea)
  }

  const updateState = (state) => {
    var status = ''
    switch (state) {
      case 'in_progress':
        status = 'Research in progress...'
        setReportActionsStatus('disabled')
        break
      case 'finished':
        status = 'Research finished!'
        setReportActionsStatus('enabled')
        break
      case 'error':
        status = 'Research failed!'
        setReportActionsStatus('disabled')
        break
      case 'initial':
        status = ''
        setReportActionsStatus('hidden')
        break
      default:
        setReportActionsStatus('disabled')
    }
    document.getElementById('status').innerHTML = status
    if (document.getElementById('status').innerHTML == '') {
      document.getElementById('status').style.display = 'none'
    } else {
      document.getElementById('status').style.display = 'block'
    }
  }

  /**
   * Shows or hides the download and copy buttons
   * @param {str} status Kind of hacky. Takes "enabled", "disabled", or "hidden". "Hidden is same as disabled but also hides the div"
   */
  const setReportActionsStatus = (status) => {
    const reportActions = document.getElementById('reportActions')
    // Disable everything in reportActions until research is finished

    if (status == 'enabled') {
      reportActions.querySelectorAll('a').forEach((link) => {
        link.classList.remove('disabled')
        link.removeAttribute('onclick')
        reportActions.style.display = 'block'
      })
    } else {
      reportActions.querySelectorAll('a').forEach((link) => {
        link.classList.add('disabled')
        link.setAttribute('onclick', 'return false;')
      })
      if (status == 'hidden') {
        reportActions.style.display = 'none'
      }
    }
  }

  const tagsInput = document.getElementById('tags-input');
  const input = document.getElementById('custom_source');

  const tags = [];

  const addTag = (url) => {
    if (tags.includes(url)) return;
    tags.push(url);

    const tagElement = document.createElement('span');
    tagElement.className = 'tag';
    tagElement.textContent = url;

    const removeButton = document.createElement('span');
    removeButton.className = 'remove-tag';
    removeButton.textContent = 'x';
    removeButton.onclick = function () {
        tagsInput.removeChild(tagElement);
        tags.splice(tags.indexOf(url), 1);
    };

    tagElement.appendChild(removeButton);
    tagsInput.insertBefore(tagElement, input);
  }

  document.addEventListener('DOMContentLoaded', init)
  return {
    startResearch,
    copyToClipboard,
    changeSource,
      addTag,
  }
})()


# D:\Documents\Github\gpt-researcher-wes\tests\documents-report-source.py

import os
import asyncio
import pytest
# Ensure this path is correct
from gpt_researcher.master.agent import GPTResearcher
from dotenv import load_dotenv
load_dotenv()

# Define the report types to test
report_types = [
    "research_report",
    "custom_report",
    "subtopic_report",
    "summary_report",
    "detailed_report",
    "quick_report"
]

# Define a common query and sources for testing
query = "What can you tell me about myself based on my documents?"

# Define the output directory
output_dir = "./outputs"


@pytest.mark.asyncio
@pytest.mark.parametrize("report_type", report_types)
async def test_gpt_researcher(report_type):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create an instance of GPTResearcher with report_source set to "documents"
    researcher = GPTResearcher(
        query=query, report_type=report_type, report_source="documents")

    # Conduct research and write the report
    await researcher.conduct_research()
    report = await researcher.write_report()

    # Define the expected output filenames
    pdf_filename = os.path.join(output_dir, f"{report_type}.pdf")
    docx_filename = os.path.join(output_dir, f"{report_type}.docx")

    # Check if the PDF and DOCX files are created
    # assert os.path.exists(pdf_filename), f"PDF file not found for report type: {report_type}"
    # assert os.path.exists(docx_filename), f"DOCX file not found for report type: {report_type}"

    # Clean up the generated files (optional)
    # os.remove(pdf_filename)
    # os.remove(docx_filename)

if __name__ == "__main__":
    pytest.main()


# D:\Documents\Github\gpt-researcher-wes\docs\docs\examples\hybrid_research.md

# Hybrid Research

## Introduction

GPT Researcher can combine web search capabilities with local document analysis to provide comprehensive, context-aware research results. 

This guide will walk you through the process of setting up and running hybrid research using GPT Researcher.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.10 or higher installed on your system
- pip (Python package installer)
- An OpenAI API key (you can also choose other supported [LLMs](../gpt-researcher/llms/llms.md))
- A Tavily API key (you can also choose other supported [Retrievers](../gpt-researcher/search-engines/retrievers.md))

## Installation

```bash
pip install gpt-researcher
```

## Setting Up the Environment

Export your API keys as environment variables:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
export TAVILY_API_KEY=your_tavily_api_key_here
```

Alternatively, you can set these in your Python script:

```python
import os
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key_here'
os.environ['TAVILY_API_KEY'] = 'your_tavily_api_key_here'
```

## Preparing Local Documents

1. Create a directory named `my-docs` in your project folder.
2. Place all relevant local documents (PDFs, TXTs, DOCXs, etc.) in this directory.

## Running Hybrid Research

Here's a basic script to run hybrid research:

```python
from gpt_researcher import GPTResearcher
import asyncio

async def get_research_report(query: str, report_type: str, report_source: str) -> str:
    researcher = GPTResearcher(query=query, report_type=report_type, report_source=report_source)
    research = await researcher.conduct_research()
    report = await researcher.write_report()
    return report

if __name__ == "__main__":
    query = "How does our product roadmap compare to emerging market trends in our industry?"
    report_source = "hybrid"

    report = asyncio.run(get_research_report(query=query, report_type="research_report", report_source=report_source))
    print(report)
```

To run the script:

1. Save it as `run_research.py`
2. Execute it with: `python run_research.py`

## Understanding the Results

The output will be a comprehensive research report that combines insights from both web sources and your local documents. The report typically includes an executive summary, key findings, detailed analysis, comparisons between your internal data and external trends, and recommendations based on the combined insights.

## Troubleshooting

1. **API Key Issues**: Ensure your API keys are correctly set and have the necessary permissions.
2. **Document Loading Errors**: Check that your local documents are in supported formats and are not corrupted.
3. **Memory Issues**: For large documents or extensive research, you may need to increase your system's available memory or adjust the `chunk_size` in the document processing step.

## FAQ

**Q: How long does a typical research session take?**
A: The duration varies based on the complexity of the query and the amount of data to process. It can range from 1-5 minutes for very comprehensive research.

**Q: Can I use GPT Researcher with other language models?**
A: Currently, GPT Researcher is optimized for OpenAI's models. Support for other models can be found [here](../gpt-researcher/llms/llms.md).

**Q: How does GPT Researcher handle conflicting information between local and web sources?**
A: The system attempts to reconcile differences by providing context and noting discrepancies in the final report. It prioritizes more recent or authoritative sources when conflicts arise.

**Q: Is my local data sent to external servers during the research process?**
A: No, your local documents are processed on your machine. Only the generated queries and synthesized information (not raw data) are sent to external services for web research.

For more information and updates, please visit the [GPT Researcher GitHub repository](https://github.com/assafelovic/gpt-researcher).

# D:\Documents\Github\gpt-researcher-wes\docs\blog\2023-09-22-gpt-researcher\index.md

---
slug: building-gpt-researcher
title: How we built GPT Researcher
authors: [assafe]
tags: [gpt-researcher, autonomous-agent, opensource, github]
---

After [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) was published, we immediately took it for a spin. The first use case that came to mind was autonomous online research. Forming objective conclusions for manual research tasks can take time, sometimes weeks, to find the right resources and information. Seeing how well AutoGPT created tasks and executed them got me thinking about the great potential of using AI to conduct comprehensive research and what it meant for the future of online research.

But the problem with AutoGPT was that it usually ran into never-ending loops, required human interference for almost every step, constantly lost track of its progress, and almost never actually completed the task.

Nonetheless, the information and context gathered during the research task were lost (such as keeping track of sources), and sometimes hallucinated.

The passion for leveraging AI for online research and the limitations I found put me on a mission to try and solve it while sharing my work with the world. This is when I created [GPT Researcher](https://github.com/assafelovic/gpt-researcher) — an open source autonomous agent for online comprehensive research.

In this article, we will share the steps that guided me toward the proposed solution.

### Moving from infinite loops to deterministic results
The first step in solving these issues was to seek a more deterministic solution that could ultimately guarantee completing any research task within a fixed time frame, without human interference.

This is when we stumbled upon the recent paper [Plan and Solve](https://arxiv.org/abs/2305.04091). The paper aims to provide a better solution for the challenges stated above. The idea is quite simple and consists of two components: first, devising a plan to divide the entire task into smaller subtasks and then carrying out the subtasks according to the plan.

![Planner-Excutor-Model](./planner.jpeg)

As it relates to research, first create an outline of questions to research related to the task, and then deterministically execute an agent for every outline item. This approach eliminates the uncertainty in task completion by breaking the agent steps into a deterministic finite set of tasks. Once all tasks are completed, the agent concludes the research.

Following this strategy has improved the reliability of completing research tasks to 100%. Now the challenge is, how to improve quality and speed?

### Aiming for objective and unbiased results
The biggest challenge with LLMs is the lack of factuality and unbiased responses caused by hallucinations and out-of-date training sets (GPT is currently trained on datasets from 2021). But the irony is that for research tasks, it is crucial to optimize for these exact two criteria: factuality and bias.

To tackle this challenges, we assumed the following:

- Law of large numbers — More content will lead to less biased results. Especially if gathered properly.
- Leveraging LLMs for the summarization of factual information can significantly improve the overall better factuality of results.

After experimenting with LLMs for quite some time, we can say that the areas where foundation models excel are in the summarization and rewriting of given content. So, in theory, if LLMs only review given content and summarize and rewrite it, potentially it would reduce hallucinations significantly.

In addition, assuming the given content is unbiased, or at least holds opinions and information from all sides of a topic, the rewritten result would also be unbiased. So how can content be unbiased? The [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers). In other words, if enough sites that hold relevant information are scraped, the possibility of biased information reduces greatly. So the idea would be to scrape just enough sites together to form an objective opinion on any topic.

Great! Sounds like, for now, we have an idea for how to create both deterministic, factual, and unbiased results. But what about the speed problem?

### Speeding up the research process
Another issue with AutoGPT is that it works synchronously. The main idea of it is to create a list of tasks and then execute them one by one. So if, let’s say, a research task requires visiting 20 sites, and each site takes around one minute to scrape and summarize, the overall research task would take a minimum of +20 minutes. That’s assuming it ever stops. But what if we could parallelize agent work?

By levering Python libraries such as asyncio, the agent tasks have been optimized to work in parallel, thus significantly reducing the time to research.

```python
# Create a list to hold the coroutine agent tasks
tasks = [async_browse(url, query, self.websocket) for url in await new_search_urls]

# Gather the results as they become available
responses = await asyncio.gather(*tasks, return_exceptions=True)
```

In the example above, we trigger scraping for all URLs in parallel, and only once all is done, continue with the task. Based on many tests, an average research task takes around three minutes (!!). That’s 85% faster than AutoGPT.

### Finalizing the research report
Finally, after aggregating as much information as possible about a given research task, the challenge is to write a comprehensive report about it.

After experimenting with several OpenAI models and even open source, I’ve concluded that the best results are currently achieved with GPT-4. The task is straightforward — provide GPT-4 as context with all the aggregated information, and ask it to write a detailed report about it given the original research task.

The prompt is as follows:
```commandline
"{research_summary}" Using the above information, answer the following question or topic: "{question}" in a detailed report — The report should focus on the answer to the question, should be well structured, informative, in depth, with facts and numbers if available, a minimum of 1,200 words and with markdown syntax and apa format. Write all source urls at the end of the report in apa format. You should write your report only based on the given information and nothing else.
```

The results are quite impressive, with some minor hallucinations in very few samples, but it’s fair to assume that as GPT improves over time, results will only get better.

### The final architecture
Now that we’ve reviewed the necessary steps of GPT Researcher, let’s break down the final architecture, as shown below:

<div align="center">
<img align="center" height="500" src="https://cowriter-images.s3.amazonaws.com/architecture.png"/>
</div>

More specifically:
- Generate an outline of research questions that form an objective opinion on any given task.
- For each research question, trigger a crawler agent that scrapes online resources for information relevant to the given task.
- For each scraped resource, keep track, filter, and summarize only if it includes relevant information.
- Finally, aggregate all summarized sources and generate a final research report.

### Going forward
The future of online research automation is heading toward a major disruption. As AI continues to improve, it is only a matter of time before AI agents can perform comprehensive research tasks for any of our day-to-day needs. AI research can disrupt areas of finance, legal, academia, health, and retail, reducing our time for each research by 95% while optimizing for factual and unbiased reports within an influx and overload of ever-growing online information.

Imagine if an AI can eventually understand and analyze any form of online content — videos, images, graphs, tables, reviews, text, audio. And imagine if it could support and analyze hundreds of thousands of words of aggregated information within a single prompt. Even imagine that AI can eventually improve in reasoning and analysis, making it much more suitable for reaching new and innovative research conclusions. And that it can do all that in minutes, if not seconds.

It’s all a matter of time and what [GPT Researcher](https://github.com/assafelovic/gpt-researcher) is all about.


# D:\Documents\Github\gpt-researcher-wes\docs\blog\2024-09-7-hybrid-research\index.md

---
slug: gptr-hybrid
title: The Future of Research is Hybrid
authors: [assafe]
tags: [hybrid-research, gpt-researcher, langchain, langgraph, tavily]
image: https://miro.medium.com/v2/resize:fit:1400/1*NgVIlZVSePqrK5EkB1wu4Q.png
---
![Hyrbrid Research with GPT Researcher](https://miro.medium.com/v2/resize:fit:1400/1*MaauY1ecsD05nL8JqW0Zdg.jpeg)

Over the past few years, we've seen an explosion of new AI tools designed to disrupt research. Some, like [ChatPDF](https://www.chatpdf.com/) and [Consensus](https://consensus.app), focus on extracting insights from documents. Others, such as [Perplexity](https://www.perplexity.ai/), excel at scouring the web for information. But here's the thing: none of these tools combine both web and local document search within a single contextual research pipeline.

This is why I'm excited to introduce the latest advancements of **[GPT Researcher](https://gptr.dev)** — now able to conduct hybrid research on any given task and documents.

Web driven research often lacks specific context, risks information overload, and may include outdated or unreliable data. On the flip side, local driven research is limited to historical data and existing knowledge, potentially creating organizational echo chambers and missing out on crucial market trends or competitor moves. Both approaches, when used in isolation, can lead to incomplete or biased insights, hampering your ability to make fully informed decisions.

Today, we're going to change the game. By the end of this guide, you'll learn how to conduct hybrid research that combines the best of both worlds — web and local — enabling you to conduct more thorough, relevant, and insightful research.

## Why Hybrid Research Works Better

By combining web and local sources, hybrid research addresses these limitations and offers several key advantages:

1. **Grounded context**: Local documents provide a foundation of verified, organization specific information. This grounds the research in established knowledge, reducing the risk of straying from core concepts or misinterpreting industry specific terminology.
   
   *Example*: A pharmaceutical company researching a new drug development opportunity can use its internal research papers and clinical trial data as a base, then supplement this with the latest published studies and regulatory updates from the web.

2. **Enhanced accuracy**: Web sources offer up-to-date information, while local documents provide historical context. This combination allows for more accurate trend analysis and decision-making.
   
   *Example*: A financial services firm analyzing market trends can combine their historical trading data with real-time market news and social media sentiment analysis to make more informed investment decisions.

3. **Reduced bias**: By drawing from both web and local sources, we mitigate the risk of bias that might be present in either source alone.
   
   *Example*: A tech company evaluating its product roadmap can balance internal feature requests and usage data with external customer reviews and competitor analysis, ensuring a well-rounded perspective.

4. **Improved planning and reasoning**: LLMs can leverage the context from local documents to better plan their web research strategies and reason about the information they find online.
   
   *Example*: An AI-powered market research tool can use a company's past campaign data to guide its web search for current marketing trends, resulting in more relevant and actionable insights.

5. **Customized insights**: Hybrid research allows for the integration of proprietary information with public data, leading to unique, organization-specific insights.
   
   *Example*: A retail chain can combine its sales data with web-scraped competitor pricing and economic indicators to optimize its pricing strategy in different regions.

These are just a few examples for business use cases that can leverage hybrid research, but enough with the small talk — let's build!

## Building the Hybrid Research Assistant

Before we dive into the details, it's worth noting that GPT Researcher has the capability to conduct hybrid research out of the box! However, to truly appreciate how this works and to give you a deeper understanding of the process, we're going to take a look under the hood.

![GPT Researcher hybrid research](./gptr-hybrid.png)

GPT Researcher conducts web research based on an auto-generated plan from local documents, as seen in the architecture above. It then retrieves relevant information from both local and web data for the final research report.

We'll explore how local documents are processed using LangChain, which is a key component of GPT Researcher's document handling. Then, we'll show you how to leverage GPT Researcher to conduct hybrid research, combining the advantages of web search with your local document knowledge base.

### Processing Local Documents with Langchain

LangChain provides a variety of document loaders that allow us to process different file types. This flexibility is crucial when dealing with diverse local documents. Here's how to set it up:

```python
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    TextLoader, 
    UnstructuredCSVLoader, 
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader, 
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def load_local_documents(file_paths):
    documents = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loader = PyMuPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = UnstructuredCSVLoader(file_path)
        elif file_path.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path)
        elif file_path.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_path.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        documents.extend(loader.load())
    
    return documents

# Use the function to load your local documents
local_docs = load_local_documents(['company_report.pdf', 'meeting_notes.docx', 'data.csv'])

# Split the documents into smaller chunks for more efficient processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(local_docs)

# Create embeddings and store them in a vector database for quick retrieval
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Example of how to perform a similarity search
query = "What were the key points from our last strategy meeting?"
relevant_docs = vectorstore.similarity_search(query, k=3)

for doc in relevant_docs:
    print(doc.page_content)
```

### Conducting Web Research with GPT Researcher

Now that we've learned how to work with local documents, let's take a quick look at how GPT Researcher works under the hood:

![GPT Researcher Architecture](https://miro.medium.com/v2/resize:fit:1400/1*yFtT43N0GxL0TMKvjtYjug.png)

As seen above, GPT Researcher creates a research plan based on the given task by generating potential research queries that can collectively provide an objective and broad overview of the topic. Once these queries are generated, GPT Researcher uses a search engine like Tavily to find relevant results. Each scraped result is then saved in a vector database. Finally, the top k chunks most related to the research task are retrieved to generate a final research report.

GPT Researcher supports hybrid research, which involves an additional step of chunking local documents (implemented using Langchain) before retrieving the most related information. After numerous evaluations conducted by the community, we've found that hybrid research improved the correctness of final results by over 40%!

### Running the Hybrid Research with GPT Researcher

Now that you have a better understanding of how hybrid research works, let's demonstrate how easy this can be achieved with GPT Researcher.

#### Step 1: Install GPT Researcher with PIP

```bash
pip install gpt-researcher
```

#### Step 2: Setting up the environment

We will run GPT Researcher with OpenAI as the LLM vendor and Tavily as the search engine. You'll need to obtain API keys for both before moving forward. Then, export the environment variables in your CLI as follows:

```bash
export OPENAI_API_KEY={your-openai-key}
export TAVILY_API_KEY={your-tavily-key}
```

#### Step 3: Initialize GPT Researcher with hybrid research configuration

GPT Researcher can be easily initialized with params that signal it to run a hybrid research. You can conduct many forms of research, head to the documentation page to learn more.

To get GPT Researcher to run a hybrid research, you need to include all relevant files in my-docs directory (create it if it doesn't exist), and set the instance report_source to "hybrid" as seen below. Once the report source is set to hybrid, GPT Researcher will look for existing documents in the my-docs directory and include them in the research. If no documents exist, it will ignore it.

```python
from gpt_researcher import GPTResearcher
import asyncio

async def get_research_report(query: str, report_type: str, report_source: str) -> str:
    researcher = GPTResearcher(query=query, report_type=report_type, report_source=report_source)
    research = await researcher.conduct_research()
    report = await researcher.write_report()
    return report
    
if __name__ == "__main__":
    query = "How does our product roadmap compare to emerging market trends in our industry?"
    report_source = "hybrid"

    report = asyncio.run(get_research_report(query=query, report_type="research_report", report_source=report_source))
    print(report)
```

As seen above, we can run the research on the following example:

- Research task: "How does our product roadmap compare to emerging market trends in our industry?"
- Web: Current market trends, competitor announcements, and industry forecasts
- Local: Internal product roadmap documents and feature prioritization lists

After various community evaluations we've found that the results of this research improve quality and correctness of research by over 40% and remove hallucinations by 50%. Moreover as stated above, local information helps the LLM improve planning reasoning allowing it to make better decisions and researching more relevant web sources.

But wait, there's more! GPT Researcher also includes a sleek front-end app using NextJS and Tailwind. To learn how to get it running check out the documentation page. You can easily use drag and drop for documents to run hybrid research.

## Conclusion

Hybrid research represents a significant advancement in data gathering and decision making. By leveraging tools like [GPT Researcher](https://gptr.dev), teams can now conduct more comprehensive, context-aware, and actionable research. This approach addresses the limitations of using web or local sources in isolation, offering benefits such as grounded context, enhanced accuracy, reduced bias, improved planning and reasoning, and customized insights.

The automation of hybrid research can enable teams to make faster, more data-driven decisions, ultimately enhancing productivity and offering a competitive advantage in analyzing an expanding pool of unstructured and dynamic information.

# D:\Documents\Github\gpt-researcher-wes\multi_agents\README.md

# LangGraph x GPT Researcher
[LangGraph](https://python.langchain.com/docs/langgraph) is a library for building stateful, multi-actor applications with LLMs. 
This example uses Langgraph to automate the process of an in depth research on any given topic.

## Use case
By using Langgraph, the research process can be significantly improved in depth and quality by leveraging multiple agents with specialized skills. 
Inspired by the recent [STORM](https://arxiv.org/abs/2402.14207) paper, this example showcases how a team of AI agents can work together to conduct research on a given topic, from planning to publication.

An average run generates a 5-6 page research report in multiple formats such as PDF, Docx and Markdown.

Please note: This example uses the OpenAI API only for optimized performance.

## The Multi Agent Team
The research team is made up of 8 agents:
- **Human** - The human in the loop that oversees the process and provides feedback to the agents.
- **Chief Editor** - Oversees the research process and manages the team. This is the "master" agent that coordinates the other agents using Langgraph.
- **Researcher** (gpt-researcher) - A specialized autonomous agent that conducts in depth research on a given topic.
- **Editor** - Responsible for planning the research outline and structure.
- **Reviewer** - Validates the correctness of the research results given a set of criteria.
- **Revisor** - Revises the research results based on the feedback from the reviewer.
- **Writer** - Responsible for compiling and writing the final report.
- **Publisher** - Responsible for publishing the final report in various formats.

## How it works
Generally, the process is based on the following stages: 
1. Planning stage
2. Data collection and analysis
3. Review and revision
4. Writing and submission
5. Publication

### Architecture
<div align="center">
<img align="center" height="600" src="https://github.com/user-attachments/assets/ef561295-05f4-40a8-a57d-8178be687b18">
</div>
<br clear="all"/>

### Steps
More specifically (as seen in the architecture diagram) the process is as follows:
- Browser (gpt-researcher) - Browses the internet for initial research based on the given research task.
- Editor - Plans the report outline and structure based on the initial research.
- For each outline topic (in parallel):
  - Researcher (gpt-researcher) - Runs an in depth research on the subtopics and writes a draft.
  - Reviewer - Validates the correctness of the draft given a set of criteria and provides feedback.
  - Revisor - Revises the draft until it is satisfactory based on the reviewer feedback.
- Writer - Compiles and writes the final report including an introduction, conclusion and references section from the given research findings.
- Publisher - Publishes the final report to multi formats such as PDF, Docx, Markdown, etc.

## How to run
1. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Update env variables
   ```bash
   export OPENAI_API_KEY={Your OpenAI API Key here}
   export TAVILY_API_KEY={Your Tavily API Key here}
   ```
2. Run the application:
    ```bash
    python main.py
    ```

## Usage
To change the research query and customize the report, edit the `task.json` file in the main directory.
#### Task.json contains the following fields:
- `query` - The research query or task.
- `model` - The OpenAI LLM to use for the agents.
- `max_sections` - The maximum number of sections in the report. Each section is a subtopic of the research query.
- `include_human_feedback` - If true, the user can provide feedback to the agents. If false, the agents will work autonomously.
- `publish_formats` - The formats to publish the report in. The reports will be written in the `output` directory.
- `source` - The location from which to conduct the research. Options: `web` or `local`. For local, please add `DOC_PATH` env var.
- `follow_guidelines` - If true, the research report will follow the guidelines below. It will take longer to complete. If false, the report will be generated faster but may not follow the guidelines.
- `guidelines` - A list of guidelines that the report must follow.
- `verbose` - If true, the application will print detailed logs to the console.

#### For example:
```json
{
  "query": "Is AI in a hype cycle?",
  "model": "gpt-4o",
  "max_sections": 3, 
  "publish_formats": { 
    "markdown": true,
    "pdf": true,
    "docx": true
  },
  "include_human_feedback": false,
  "source": "web",
  "follow_guidelines": true,
  "guidelines": [
    "The report MUST fully answer the original question",
    "The report MUST be written in apa format",
    "The report MUST be written in english"
  ],
  "verbose": true
}
```

## To Deploy

```shell
pip install langgraph-cli
langgraph up
```

From there, see documentation [here](https://github.com/langchain-ai/langgraph-example) on how to use the streaming and async endpoints, as well as the playground.


# D:\Documents\Github\gpt-researcher-wes\docs\docs\gpt-researcher\gptr\config.md

# Configuration

The config.py enables you to customize GPT Researcher to your specific needs and preferences.

Thanks to our amazing community and contributions, GPT Researcher supports multiple LLMs and Retrievers.
In addition, GPT Researcher can be tailored to various report formats (such as APA), word count, research iterations depth, etc.

GPT Researcher defaults to our recommended suite of integrations: [OpenAI](https://platform.openai.com/docs/overview) for LLM calls and [Tavily API](https://app.tavily.com) for retrieving real-time web information.

As seen below, OpenAI still stands as the superior LLM. We assume it will stay this way for some time, and that prices will only continue to decrease, while performance and speed increase over time.

<div style={{ marginBottom: '10px' }}>
<img align="center" height="350" src="/img/leaderboard.png" />
</div>

The default config.py file can be found in `/gpt_researcher/config/`. It supports various options for customizing GPT Researcher to your needs.
You can also include your own external JSON file `config.json` by adding the path in the `config_file` param. **Please follow the config.py file for additional future support**.

Below is a list of current supported options:

- **`RETRIEVER`**: Web search engine used for retrieving sources. Defaults to `tavily`. Options: `duckduckgo`, `bing`, `google`, `searchapi`, `serper`, `searx`. [Check here](https://github.com/assafelovic/gpt-researcher/tree/master/gpt_researcher/retrievers) for supported retrievers
- **`EMBEDDING_PROVIDER`**: Provider for embedding model. Defaults to `openai`. Options: `ollama`, `huggingface`, `azure_openai`, `custom`.
- **`FAST_LLM`**: Model name for fast LLM operations such summaries. Defaults to `openai:gpt-4o-mini`.
- **`SMART_LLM`**: Model name for smart operations like generating research reports and reasoning. Defaults to `openai:gpt-4o`.
- **`FAST_TOKEN_LIMIT`**: Maximum token limit for fast LLM responses. Defaults to `2000`.
- **`SMART_TOKEN_LIMIT`**: Maximum token limit for smart LLM responses. Defaults to `4000`.
- **`BROWSE_CHUNK_MAX_LENGTH`**: Maximum length of text chunks to browse in web sources. Defaults to `8192`.
- **`SUMMARY_TOKEN_LIMIT`**: Maximum token limit for generating summaries. Defaults to `700`.
- **`TEMPERATURE`**: Sampling temperature for LLM responses, typically between 0 and 1. A higher value results in more randomness and creativity, while a lower value results in more focused and deterministic responses. Defaults to `0.55`.
- **`TOTAL_WORDS`**: Total word count limit for document generation or processing tasks. Defaults to `800`.
- **`REPORT_FORMAT`**: Preferred format for report generation. Defaults to `APA`. Consider formats like `MLA`, `CMS`, `Harvard style`, `IEEE`, etc.
- **`MAX_ITERATIONS`**: Maximum number of iterations for processes like query expansion or search refinement. Defaults to `3`.
- **`AGENT_ROLE`**: Role of the agent. This might be used to customize the behavior of the agent based on its assigned roles. No default value.
- **`MAX_SUBTOPICS`**: Maximum number of subtopics to generate or consider. Defaults to `3`.
- **`SCRAPER`**: Web scraper to use for gathering information. Defaults to `bs` (BeautifulSoup). You can also use [newspaper](https://github.com/codelucas/newspaper).
- **`DOC_PATH`**: Path to read and research local documents. Defaults to an empty string indicating no path specified.
- **`USER_AGENT`**: Custom User-Agent string for web crawling and web requests.
- **`MEMORY_BACKEND`**: Backend used for memory operations, such as local storage of temporary data. Defaults to `local`.

To change the default configurations, you can simply add env variables to your `.env` file as named above or export manually in your local project directory.

For example, to manually change the search engine and report format:
```bash
export RETRIEVER=bing
export REPORT_FORMAT=IEEE
```
Please note that you might need to export additional env vars and obtain API keys for other supported search retrievers and LLM providers. Please follow your console logs for further assistance.
To learn more about additional LLM support you can check out the docs [here](/docs/gpt-researcher/llms/llms).

You can also include your own external JSON file `config.json` by adding the path in the `config_file` param.

## Example: Azure OpenAI Configuration

If you are not using OpenAI's models, but other model providers, besides the general configuration above, also additional environment variables are required.
Check the [langchain documentation](https://python.langchain.com/v0.2/docs/integrations/platforms/) about your model for the exact configuration of the API keys and endpoints.

Here is an example for [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models) configuration:

```bash

OPENAI_API_VERSION="2024-05-01-preview" # or whatever you are using
AZURE_OPENAI_ENDPOINT="https://CHANGEMEN.openai.azure.com/" # change to the name of your deployment
AZURE_OPENAI_API_KEY="CHANGEME" # change to your API key

EMBEDDING_PROVIDER="azureopenai"
AZURE_EMBEDDING_MODEL="text-embedding-ada-002" # change to the deployment of your embedding model

FAST_LLM="azure_openai:gpt-4o-mini" # change to the name of your deployment (not model-name)
FAST_TOKEN_LIMIT=4000

SMART_LLM="azure_openai:gpt-4o" # change to the name of your deployment (not model-name)
SMART_TOKEN_LIMIT=4000

RETRIEVER="bing" # if you are using Bing as your search engine (which is likely if you use Azure)
BING_API_KEY="CHANGEME"

```


# D:\Documents\Github\gpt-researcher-wes\docs\docs\gpt-researcher\gptr\example.md

# Agent Example

If you're interested in using GPT Researcher as a standalone agent, you can easily import it into any existing Python project. Below, is an example of calling the agent to generate a research report:

```python
from gpt_researcher import GPTResearcher
import asyncio

async def fetch_report(query):
    """
    Fetch a research report based on the provided query and report type.
    """
    researcher = GPTResearcher(query=query)
    await researcher.conduct_research()
    report = await researcher.write_report()
    return report

async def generate_research_report(query):
    """
    This is a sample script that executes an async main function to run a research report.
    """
    report = await fetch_report(query)
    print(report)

if __name__ == "__main__":
    QUERY = "What happened in the latest burning man floods?"
    asyncio.run(generate_research_report(query=QUERY))
```

You can further enhance this example to use the returned report as context for generating valuable content such as news article, marketing content, email templates, newsletters, etc.

You can also use GPT Researcher to gather information about code documentation, business analysis, financial information and more. All of which can be used to complete much more complex tasks that require factual and high quality realtime information.


# D:\Documents\Github\gpt-researcher-wes\docs\docs\gpt-researcher\getting-started\how-to-choose.md

# How to Choose

GPT Researcher is a powerful autonomous research agent designed to enhance and streamline your research processes. Whether you're a developer looking to integrate research capabilities into your project or an end-user seeking a comprehensive research solution, GPT Researcher offers flexible options to meet your needs.

We envision a future where AI agents collaborate to complete complex tasks, with research being a critical step in the process. GPT Researcher aims to be your go-to agent for any research task, regardless of complexity. It can be easily integrated into existing agent workflows, eliminating the need to create your own research agent from scratch.

## Options

GPT Researcher offers multiple ways to leverage its capabilities:

<img src="https://github.com/user-attachments/assets/305fa3b9-60fa-42b6-a4b0-84740ab6c665" alt="Logo" width="568"></img>
<br></br>

1. **GPT Researcher PIP agent**: Ideal for integrating GPT Researcher into your existing projects and workflows.
2. **Backend**: A backend service to interact with the frontend user interfaces, offering advanced features like detailed reports.
3. **Multi Agent System**: An advanced setup using LangGraph, offering the most comprehensive research capabilities.
4. **Frontend**: Several front-end solutions depending on your needs, including a simple HTML/JS version and a more advanced NextJS version.

## Usage Options

### 1. PIP Package

The PIP package is ideal for leveraging GPT Researcher as an agent in your preferred environment and code.

**Pros:**
- Easy integration into existing projects
- Flexible usage in multi-agent systems, chains, or workflows
- Optimized for production performance

**Cons:**
- Requires some coding knowledge
- May need additional setup for advanced features

**Installation:**
```
pip install gpt-researcher
```

**System Requirements:**
- Python 3.10+
- pip package manager

**Learn More:** [PIP Documentation](https://docs.gptr.dev/docs/gpt-researcher/gptr/pip-package)

### 2. End-to-End Application

For a complete out-of-the-box experience, including a sleek frontend, you can clone our repository.

**Pros:**
- Ready-to-use frontend and backend services
- Includes advanced use cases like detailed report generation
- Optimal user experience

**Cons:**
- Less flexible than the PIP package for custom integrations
- Requires setting up the entire application

**Getting Started:**
1. Clone the repository: `git clone https://github.com/assafelovic/gpt-researcher.git`
2. Follow the [installation instructions](https://docs.gptr.dev/docs/gpt-researcher/getting-started/getting-started)

**System Requirements:**
- Git
- Python 3.10+
- Node.js and npm (for frontend)

**Advanced Usage Example:** [Detailed Report Implementation](https://github.com/assafelovic/gpt-researcher/tree/master/backend/report_type/detailed_report)

### 3. Multi Agent System with LangGraph

We've collaborated with LangChain to support multi-agents with LangGraph and GPT Researcher, offering the most complex and comprehensive version of GPT Researcher.

**Pros:**
- Very detailed, customized research reports
- Inner AI agent loops and reasoning

**Cons:**
- More expensive and time-consuming
- Heavyweight for production use

This version is recommended for local, experimental, and educational use. We're working on providing a lighter version soon!

**System Requirements:**
- Python 3.10+
- LangGraph library

**Learn More:** [GPT Researcher x LangGraph](https://docs.gptr.dev/docs/gpt-researcher/multi_agents/langgraph)

## Comparison Table

| Feature | PIP Package | End-to-End Application | Multi Agent System |
|---------|-------------|------------------------|---------------------|
| Ease of Integration | High | Medium | Low |
| Customization | High | Medium | High |
| Out-of-the-box UI | No | Yes | No |
| Complexity | Low | Medium | High |
| Best for | Developers | End-users | Researchers/Experimenters |

Please note that all options have been optimized and refined for production use.

## Deep Dive

To learn more about each of the options, check out these docs and code snippets:

1. **PIP Package**: 
   - Install: `pip install gpt-researcher`
   - [Integration guide](https://docs.gptr.dev/docs/gpt-researcher/gptr/pip-package)

2. **End-to-End Application**: 
   - Clone the repository: `git clone https://github.com/assafelovic/gpt-researcher.git`
   - [Installation instructions](https://docs.gptr.dev/docs/gpt-researcher/getting-started/getting-started)

3. **Multi-Agent System**: 
   - [Multi-Agents code](https://github.com/assafelovic/gpt-researcher/tree/master/multi_agents)
   - [LangGraph documentation](https://docs.gptr.dev/docs/gpt-researcher/multi_agents/langgraph)
   - [Blog](https://docs.gptr.dev/blog/gptr-langgraph)

## Versioning and Updates

GPT Researcher is actively maintained and updated. To ensure you're using the latest version:

- For the PIP package: `pip install --upgrade gpt-researcher`
- For the End-to-End Application: Pull the latest changes from the GitHub repository
- For the Multi-Agent System: Check the documentation for compatibility with the latest LangChain and LangGraph versions

## Troubleshooting and FAQs

For common issues and questions, please refer to our [FAQ section](https://docs.gptr.dev/docs/faq) in the documentation.


# D:\Documents\Github\gpt-researcher-wes\docs\docs\gpt-researcher\getting-started\introduction.md

# Introduction

[![Official Website](https://img.shields.io/badge/Official%20Website-gptr.dev-teal?style=for-the-badge&logo=world&logoColor=white)](https://gptr.dev)
[![Discord Follow](https://dcbadge.vercel.app/api/server/QgZXvJAccX?style=for-the-badge&theme=clean-inverted)](https://discord.gg/QgZXvJAccX)

[![GitHub Repo stars](https://img.shields.io/github/stars/assafelovic/gpt-researcher?style=social)](https://github.com/assafelovic/gpt-researcher)
[![Twitter Follow](https://img.shields.io/twitter/follow/assaf_elovic?style=social)](https://twitter.com/assaf_elovic)
[![PyPI version](https://badge.fury.io/py/gpt-researcher.svg)](https://badge.fury.io/py/gpt-researcher)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/assafelovic/gpt-researcher/blob/master/docs/docs/examples/pip-run.ipynb)

**[GPT Researcher](https://gptr.dev) is an autonomous agent designed for comprehensive online research on a variety of tasks.** 

The agent can produce detailed, factual and unbiased research reports, with customization options for focusing on relevant resources, outlines, and lessons. Inspired by the recent [Plan-and-Solve](https://arxiv.org/abs/2305.04091) and [RAG](https://arxiv.org/abs/2005.11401) papers, GPT Researcher addresses issues of speed, determinism and reliability, offering a more stable performance and increased speed through parallelized agent work, as opposed to synchronous operations.

## Why GPT Researcher?

- To form objective conclusions for manual research tasks can take time, sometimes weeks to find the right resources and information.
- Current LLMs are trained on past and outdated information, with heavy risks of hallucinations, making them almost irrelevant for research tasks.
- Current LLMs are limited to short token outputs which are not sufficient for long detailed research reports (2k+ words).
- Solutions that enable web search (such as ChatGPT + Web Plugin), only consider limited resources and content that in some cases result in superficial conclusions or biased answers.
- Using only a selection of resources can create bias in determining the right conclusions for research questions or tasks. 

## Architecture
The main idea is to run "planner" and "execution" agents, whereas the planner generates questions to research, and the execution agents seek the most related information based on each generated research question. Finally, the planner filters and aggregates all related information and creates a research report. <br /> <br /> 
The agents leverage both gpt-4o-mini and gpt-4o (128K context) to complete a research task. We optimize for costs using each only when necessary. **The average research task takes around 3 minutes to complete, and costs ~$0.1.**

<div align="center">
<img align="center" height="600" src="https://github.com/assafelovic/gpt-researcher/assets/13554167/4ac896fd-63ab-4b77-9688-ff62aafcc527" />
</div>


More specifically:
* Create a domain specific agent based on research query or task.
* Generate a set of research questions that together form an objective opinion on any given task. 
* For each research question, trigger a crawler agent that scrapes online resources for information relevant to the given task.
* For each scraped resources, summarize based on relevant information and keep track of its sources.
* Finally, filter and aggregate all summarized sources and generate a final research report.

## Demo
<iframe height="400" width="700" src="https://github.com/assafelovic/gpt-researcher/assets/13554167/a00c89a6-a295-4dd0-b58d-098a31c40fda" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## Tutorials
 - [How it Works](https://medium.com/better-programming/how-i-built-an-autonomous-ai-agent-for-online-research-93435a97c6c)
 - [How to Install](https://www.loom.com/share/04ebffb6ed2a4520a27c3e3addcdde20?sid=da1848e8-b1f1-42d1-93c3-5b0b9c3b24ea)
 - [Live Demo](https://www.loom.com/share/6a3385db4e8747a1913dd85a7834846f?sid=a740fd5b-2aa3-457e-8fb7-86976f59f9b8)
 - [Homepage](https://gptr.dev)

## Features
- 📝 Generate research, outlines, resources and lessons reports
- 📜 Can generate long and detailed research reports (over 2K words)
- 🌐 Aggregates over 20 web sources per research to form objective and factual conclusions
- 🖥️ Includes an easy-to-use web interface (HTML/CSS/JS)
- 🔍 Scrapes web sources with javascript support
- 📂 Keeps track and context of visited and used web sources
- 📄 Export research reports to PDF, Word and more...

Let's get started [here](/docs/gpt-researcher/getting-started/getting-started)!
