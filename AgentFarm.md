# Enhancing the GPT Researcher App

This guide provides comprehensive steps to enhance the GPT Researcher app by adding new modular agent types such as Legal Assistant, Financial Assistant, and more. It ensures the system remains extensible for future agent additions with specific prompts and functionalities.

## Table of Contents

1. [Understand the Current Agent Selection Mechanism](#1-understand-the-current-agent-selection-mechanism)
2. [Create a Modular Agent Configuration](#2-create-a-modular-agent-configuration)
   - [2.1 Create an Agent Configuration File](#21-create-an-agent-configuration-file)
   - [2.2 Load Agent Configurations in Code](#22-load-agent-configurations-in-code)
   - [2.3 Modify the `auto_agent_instructions` Function](#23-modify-the-auto_agent_instructions-function)
3. [Implement Agent Creation Logic](#3-implement-agent-creation-logic)
   - [3.1 Base Agent Class](#31-base-agent-class)
   - [3.2 Specific Agent Classes](#32-specific-agent-classes)
4. [Adjust the Agent Selection Process](#4-adjust-the-agent-selection-process)
5. [Modify the GPTResearcher Class](#5-modify-the-gpresearcher-class)
6. [Customize Search Queries and Results Processing](#6-customize-search-queries-and-results-processing)
7. [Implement Agent-Specific Report Generation](#7-implement-agent-specific-report-generation)
8. [Ensure Virtual Environment Is Activated](#8-ensure-virtual-environment-is-activated)
9. [Adding New Agents](#9-adding-new-agents)
10. [Testing and Validation](#10-testing-and-validation)
11. [Documentation and Comments](#11-documentation-and-comments)
12. [Potential Issues and Solutions](#12-potential-issues-and-solutions)
13. [Example of Using the Modified System](#13-example-of-using-the-modified-system)
14. [Guiding the Researcher to Specific Websites](#14-guiding-the-researcher-to-specific-websites)
15. [Final Thoughts](#15-final-thoughts)

---

## 1. Understand the Current Agent Selection Mechanism

The agent selection is currently handled in the `auto_agent_instructions()` function within `prompts.py`. This function provides a prompt that helps the language model map a given task to an appropriate agent with a specific role prompt.

### Existing Code
python:prompts.py
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
"agent_role_prompt": "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends."
}
task: "could reselling sneakers become profitable?"
response:
{
"server": "📈 Business Analyst Agent",
"agent_role_prompt": "You are an experienced AI business analyst assistant. Your main objective is to produce comprehensive, insightful, impartial, and systematically structured business reports based on provided business data, market trends, and strategic analysis."
}
task: "what are the most interesting sites in Tel Aviv?"
response:
{
"server": "🌍 Travel Agent",
"agent_role_prompt": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights."
}
"""


This function provides examples that the language model uses to map tasks to agents.

---

## 2. Create a Modular Agent Configuration

To make the app modular and allow easy addition of new agent types, externalize agent configurations into a JSON or YAML file. This approach enables defining agent properties without modifying the core code.

### 2.1 Create an Agent Configuration File

Create a file named `agents_config.json` to define your agents.

```json:agents_config.json
{
  "Finance Agent": {
    "emoji": "💰",
    "role_prompt": "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends.",
    "preferred_websites": ["https://www.investopedia.com", "https://www.bloomberg.com"],
    "output_format": "financial report"
  },
  "Legal Assistant": {
    "emoji": "⚖️",
    "role_prompt": "You are an expert legal assistant specializing in providing thorough, accurate, and well-structured legal research and summaries based on relevant laws, statutes, and legal precedents.",
    "preferred_websites": ["https://www.law.cornell.edu", "https://www.legalzoom.com"],
    "output_format": "legal memo"
  },
  "Medical Research Assistant": {
    "emoji": "🩺",
    "role_prompt": "You are a medical research assistant specializing in providing comprehensive, accurate, and up-to-date medical research summaries based on the latest studies, clinical trials, and medical guidelines.",
    "preferred_websites": ["https://www.ncbi.nlm.nih.gov", "https://www.who.int"],
    "output_format": "medical research report"
  }
}
```

*Add as many agents as needed, specifying their properties.*

### 2.2 Load Agent Configurations in Code

Modify your code to load the configuration from the `agents_config.json` file.

```python:utils/agent_config.py
import json
import os

def load_agent_config():
    config_path = os.path.join(os.path.dirname(__file__), 'agents_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
```

### 2.3 Modify the `auto_agent_instructions` Function

Update the `auto_agent_instructions()` function to dynamically generate examples from the configuration.

```python:prompts.py
from utils.agent_config import load_agent_config

def auto_agent_instructions():
    agent_config = load_agent_config()
    examples = ''
    for agent_name, details in agent_config.items():
        task_example = f"[Provide an example task for {agent_name}]"
        examples += f'task: "{task_example}"\nresponse:\n{{\n    "server": "{details["emoji"]} {agent_name}",\n    "agent_role_prompt": "{details["role_prompt"]}"\n}}\n'
    return f"""
This task involves researching a given topic, regardless of its complexity or the availability of a definitive answer...

examples:
{examples}
"""
```

*Ensure you replace `[Provide an example task for {agent_name}]` with actual example tasks or make this dynamic as well.*

---

## 3. Implement Agent Creation Logic

Create a factory function that creates agent instances based on the configuration.

### 3.1 Base Agent Class

```python:agents/base_agent.py
class BaseAgent:
    """
    Base class for all agents.

    Attributes:
        name (str): Name of the agent.
        emoji (str): Emoji representing the agent.
        role_prompt (str): Role prompt for the agent.
        preferred_websites (list): List of preferred websites for search.
        output_format (str): Output format for the agent's reports.
    """

    def __init__(self, config):
        self.name = config.get('name')
        self.emoji = config.get('emoji')
        self.role_prompt = config.get('role_prompt')
        self.preferred_websites = config.get('preferred_websites', [])
        self.output_format = config.get('output_format')
```

### 3.2 Specific Agent Classes

Create specific agent classes as needed, inheriting from `BaseAgent`.

```python:agents/legal_agent.py
from agents.base_agent import BaseAgent

class LegalAgent(BaseAgent):
    def generate_search_queries(self, query):
        # Implement legal-specific search query generation
        pass

    def process_results(self, results):
        # Implement legal-specific result processing
        pass
```

Similarly, create other agent classes:

```python:agents/finance_agent.py
from agents.base_agent import BaseAgent

class FinanceAgent(BaseAgent):
    def generate_search_queries(self, query):
        # Implement finance-specific search query generation
        pass

    def process_results(self, results):
        # Implement finance-specific result processing
        pass
```

```python:agents/medical_agent.py
from agents.base_agent import BaseAgent

class MedicalAgent(BaseAgent):
    def generate_search_queries(self, query):
        # Implement medical-specific search query generation
        pass

    def process_results(self, results):
        # Implement medical-specific result processing
        pass
```

---

## 4. Adjust the Agent Selection Process

Update the `choose_agent()` function in `query_processing.py` to use the new configuration.

```python:actions/query_processing.py
from utils.agent_config import load_agent_config
from agent_factory import create_agent
from agents.base_agent import BaseAgent

async def choose_agent(query, cfg, parent_query=None, cost_callback=None, headers=None):
    agent_config = load_agent_config()

    # Implement your logic to select the appropriate agent
    agent_name = select_agent_name(query)  # Define this function based on your selection criteria

    if agent_name in agent_config:
        agent_details = agent_config[agent_name]
        agent = create_agent(agent_name, agent_details)
        return agent
    else:
        # Default to a base agent or handle the error accordingly
        return BaseAgent({'name': 'Default Agent', 'role_prompt': 'You are a helpful assistant.'})
```

*You need to implement the `select_agent_name(query)` function to map a query to an agent name. This can be done using regex, keyword matching, or a small language model.*

---

## 5. Modify the GPTResearcher Class

Update the `GPTResearcher` class to utilize the agent's properties.

```python:agent/master.py
class GPTResearcher:
    def __init__(self, agent, query, cfg, ...):
        self.agent = agent
        self.query = query
        self.cfg = cfg
        # Other initialization...

    async def conduct_research(self):
        # Use the agent's preferred websites
        search_queries = self.agent.generate_search_queries(self.query)
        results = await self.perform_search(search_queries)
        processed_results = self.agent.process_results(results)
        # Continue with the research process...
```

---

## 6. Customize Search Queries and Results Processing

Implement agent-specific logic for generating search queries and processing results.

```python:agents/legal_agent.py
class LegalAgent(BaseAgent):
    def generate_search_queries(self, query):
        # Use legal databases or append site-specific queries
        queries = [f"site:{site} {query}" for site in self.preferred_websites]
        return queries

    def process_results(self, results):
        # Implement any legal-specific processing
        processed_results = []
        for result in results:
            # Process each result as needed
            processed_results.append(result)
        return processed_results
```

---

## 7. Implement Agent-Specific Report Generation

Modify report generation to use the agent's output format.

```python:actions/report_generation.py
async def generate_report(self, agent, processed_results):
    if agent.output_format == "legal memo":
        # Generate a legal memo
        report = await create_legal_memo(processed_results)
    elif agent.output_format == "financial report":
        # Generate a financial report
        report = await create_financial_report(processed_results)
    elif agent.output_format == "medical research report":
        # Generate a medical research report
        report = await create_medical_research_report(processed_results)
    else:
        # Generate a default report
        report = await create_default_report(processed_results)
    return report
```

*Ensure you define the `create_legal_memo`, `create_financial_report`, `create_medical_research_report`, and `create_default_report` functions accordingly.*

---

## 8. Ensure Virtual Environment Is Activated

Make sure you're working within an activated virtual environment to manage dependencies effectively.

### Create and Activate Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Unix or Linux)
source venv/bin/activate
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

*Always ensure the virtual environment is activated when working with Python or similar environments.*

---

## 9. Adding New Agents

To add a new agent in the future, follow these steps:

1. **Update `agents_config.json`** with the new agent's configuration.

    ```json:agents_config.json
    {
      "Data Science Assistant": {
        "emoji": "📊",
        "role_prompt": "You are a data science assistant specializing in providing detailed analyses and insights based on data trends and statistical models.",
        "preferred_websites": ["https://towardsdatascience.com", "https://www.kaggle.com"],
        "output_format": "data science report"
      }
    }
    ```

2. **Create the Corresponding Agent Class** if specific behaviors are needed.

    ```python:agents/data_science_agent.py
    from agents.base_agent import BaseAgent

    class DataScienceAgent(BaseAgent):
        def generate_search_queries(self, query):
            # Implement data science-specific search query generation
            pass

        def process_results(self, results):
            # Implement data science-specific result processing
            pass
    ```

3. **Update the Agent Factory** to include the new agent.

    ```python:agent_factory.py
    from agents.base_agent import BaseAgent
    from agents.finance_agent import FinanceAgent
    from agents.legal_agent import LegalAgent
    from agents.medical_agent import MedicalAgent
    from agents.data_science_agent import DataScienceAgent
    # Import other specific agents as necessary

    def create_agent(agent_name, agent_config):
        if agent_name == "Finance Agent":
            return FinanceAgent(agent_config)
        elif agent_name == "Legal Assistant":
            return LegalAgent(agent_config)
        elif agent_name == "Medical Research Assistant":
            return MedicalAgent(agent_config)
        elif agent_name == "Data Science Assistant":
            return DataScienceAgent(agent_config)
        else:
            return BaseAgent(agent_config)
    ```

---

## 10. Testing and Validation

Ensure the enhanced system works as expected by performing thorough testing.

### Unit Tests

Write unit tests for individual modules to ensure correctness.

```python:tests/test_agent_creation.py
import unittest
from agent_factory import create_agent
from utils.agent_config import load_agent_config

class TestAgentCreation(unittest.TestCase):
    def setUp(self):
        self.agent_config = load_agent_config()

    def test_create_finance_agent(self):
        agent = create_agent("Finance Agent", self.agent_config["Finance Agent"])
        self.assertEqual(agent.name, "Finance Agent")
        self.assertEqual(agent.emoji, "💰")

    def test_create_legal_agent(self):
        agent = create_agent("Legal Assistant", self.agent_config["Legal Assistant"])
        self.assertEqual(agent.name, "Legal Assistant")
        self.assertEqual(agent.emoji, "⚖️")

    def test_create_default_agent(self):
        agent = create_agent("Unknown Agent", {})
        self.assertEqual(agent.name, "Default Agent")
        self.assertEqual(agent.role_prompt, "You are a helpful assistant.")

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

Perform end-to-end testing with various agents and queries to ensure seamless integration.

### Debugging

Use logging and exception handling to identify and fix issues.

```python:utils/logger.py
import logging

def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
```

*Implement logging in your modules to trace the application's behavior.*

---

## 11. Documentation and Comments

Ensure all your code is well-documented to maintain clarity and ease of maintenance.

### Example: Base Agent Class Documentation

```python:agents/base_agent.py
class BaseAgent:
    """
    Base class for all agents.

    Attributes:
        name (str): Name of the agent.
        emoji (str): Emoji representing the agent.
        role_prompt (str): Role prompt for the agent.
        preferred_websites (list): List of preferred websites for search.
        output_format (str): Output format for the agent's reports.
    """

    def __init__(self, config):
        """
        Initializes the BaseAgent with the provided configuration.

        Args:
            config (dict): Configuration dictionary for the agent.
        """
        self.name = config.get('name')
        self.emoji = config.get('emoji')
        self.role_prompt = config.get('role_prompt')
        self.preferred_websites = config.get('preferred_websites', [])
        self.output_format = config.get('output_format')
```

*Apply similar documentation practices across all modules and classes.*

---

## 12. Potential Issues and Solutions

### LLM Output Variability

**Issue:** The language model might not always return consistent JSON.

**Solution:** Implement robust parsing with error handling.

```python:actions/query_processing.py
import json
import logging

async def choose_agent(query, cfg, parent_query=None, cost_callback=None, headers=None):
    try:
        # Assume get_llm_response is a function that gets the LLM's response
        response = await get_llm_response(query)
        agent_info = json.loads(response)
        agent_name = agent_info.get("server").split(" ", 1)[1]
        agent_config = load_agent_config()
        if agent_name in agent_config:
            return create_agent(agent_name, agent_config[agent_name])
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    # Fallback to default agent
    return BaseAgent({'name': 'Default Agent', 'role_prompt': 'You are a helpful assistant.'})
```

### Performance Optimization

**Issue:** Slowdowns due to synchronous operations or inefficient API calls.

**Solution:** Use asynchronous I/O operations and optimize API calls.

```python:actions/search.py
import aiohttp
import asyncio
import logging

async def perform_search(self, queries, agent):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for query in queries:
            tasks.append(fetch(session, query))
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for response in responses:
            if isinstance(response, Exception):
                logging.error(f"Search failed: {response}")
            else:
                results.append(response)
    return results

async def fetch(session, query):
    async with session.get(f"https://api.search.com/search?q={query}") as response:
        return await response.json()
```

### Scalability

**Issue:** Handling a large number of agents or high-traffic scenarios.

**Solution:** Implement caching mechanisms and consider scaling your infrastructure.

```python:utils/cache.py
import asyncio
from functools import lru_cache
from utils.agent_config import load_agent_config
from agent_factory import create_agent

@lru_cache(maxsize=128)
def get_cached_agent(agent_name):
    agent_config = load_agent_config()
    return create_agent(agent_name, agent_config.get(agent_name, {}))
```

---

## 13. Example of Using the Modified System

### User Query

```
What are the implications of the latest tax reform on small businesses?
```

### Agent Selection

- The system identifies this as a legal or financial query.
- It selects the "Finance Agent" or "Legal Assistant" based on keyword matching.

### Agent Behavior

- **Preferred Websites:** Uses `https://www.irs.gov` and other relevant sites.
- **Search Queries:** Generates queries focused on tax reforms and small businesses.
- **Report Generation:** Produces a financial report or legal memo, as specified by the agent's `output_format`.

---

## 14. Guiding the Researcher to Specific Websites

Modify the search function to prioritize or limit searches to the agent's preferred websites.

```python:actions/search.py
async def perform_search(self, queries, agent):
    results = []
    for query in queries:
        for site in agent.preferred_websites:
            site_query = f"site:{site} {query}"
            # Use your search API or scraping mechanism
            result = await search_api.search(site_query)
            results.append(result)
    return results
```

*This ensures the researcher targets specific websites relevant to the agent's purpose.*

---

## 15. Final Thoughts

By externalizing agent configurations and using a factory pattern, the codebase becomes modular and extensible. This approach allows adding new agents with minimal changes to the core logic, keeping the code organized and easier to maintain.

### Best Practices

- **Maintain Clear Documentation:** Ensure all new agents and their functionalities are well-documented.
- **Adhere to Coding Standards:** Follow consistent coding standards to enhance readability.
- **Regularly Update Dependencies:** Keep libraries and dependencies up to date to leverage new features and security patches.
- **Monitor Performance:** Continuously monitor the application's performance and optimize as needed.
- **Secure Sensitive Data:** Ensure any sensitive information accessed or processed by agents is handled securely.

If you need further assistance or have specific questions about any part of this process, feel free to reach out!
```

