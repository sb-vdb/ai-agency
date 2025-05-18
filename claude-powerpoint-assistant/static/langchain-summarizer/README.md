# Model-based Article Summarization

<a id ="model-based-article-summarization"></a>

- [Model-based Article Summarization](#model-based-article-summarization)
  - [Setup](#setup)
  - [Building the Article Summarizer Worker](#building-the-article-summarizer-worker)
    - [Model(-Factory)](#model-factory)
    - [Structured Output](#structured-output)
    - [Agentic Unit](#agentic-unit)
  - [Prepare Agents with different Models](#prepare-agents-with-different-models)
  - [Runner](#runner)
    - [HTML Retrieval](#html-retrieval)
    - [Batch Runner](#batch-runner)
    - [Unit Runner (Tool)](#unit-runner-tool)

<a id="introduction">Introduction</a>
- This case walks through creating AI applications and agents with **LangChain** and **LangGraph**.
- The use case is to take a URL, load the contents and have them extracted by a language model to create a unified summary.
- We will spin up the straight forward way to extract article content to use for development and benchmarking in this Notebook to examine the most basic pieces

This notebook aims to streamline a basic ai-induced workflow. We are simply not iterested in writing explicit logic to extract human-readable data from website data - so using an LM for that. 
While agentic would imply an LM to make core workflow decisions, our LMs here only execute a single instruction, not even a loop questioning the own output is used.

Our goal is something different: make an ai task predictable given a certain type of input. This does not much magic, it's more like we caged Dumbledore in a cell to only have him summon our same four favourite snacks.

In order to achieve this, many LMs support a concept called **Structured Output** natively. Actually, this is pretty straightforward if you interpret LMs as fill-in-the-blank engines. Usually, the blank we are providing is the empty space after our question(mark). But LMs are perfectly working with intermediate spaces to fill in the. That way you can create expressions with guarnteed patterns like we know it from String interpolation. In Python e.g. you can use curly brackets inside f""-Strings to reference variables, where its' values will be baked into the string at place at runtime. In the same way we can pass a fill-in-the-blank text, where we indicate the blanks with curly brackets and put individual sub-prompts about what to fill into these gaps.

That way we can make our output machine-usable in classic workflows. (LangChain offers a very elegant way to Python your way through this)

Later on, we might want to encapsulate this logic and actually use it inside agentic workflows. But this will not be covered by this notebook, as this will require more code and abstraction, while we simply want to play on the open heart here.

But as a special, we will encapsulate parts to make them iterable over a range of model configurations to batch-execute and compare by a single input.
Finally, we will encapsulate our agent as a single piece of work to wrap it into a tool, that you might use further, if you want to extend this notebook.

## [Setup](#model-based-article-summarization)
<a id="setup"></a>

## [Building the Article Summarizer Worker](#model-based-article-summarization)
<a id="building-the-article-summarizer-worker"></a>

### [Model(-Factory)](#model-based-article-summarization)
<a id="model-factory"></a>
For this Demo, we will only use the Ollama integration for using our local ai application. Before using it, make sure to properly setup Ollama.
But we want to enable benchmarking various Ollama models. Hence, we make a function to make it straightforward to use different models.

```python
from langchain_ollama import ChatOllama

OLLAMA_URL = "http://192.168.188.24:11434" # Change here, if your Ollama host runs on a different machine

def get_model(model_name: str, temperature: int = 0.1):
    return ChatOllama(model=model_name, temperature=temperature, base_url=OLLAMA_URL)
```

### [Structured Output](#model-based-article-summarization)
<a id="summarizer"></a>
LangChain is a huge platform to make application logic with AI components; offers a huge range of integrations and helpers to achieve all sorts of interactions. We will use a tiny subset of the features, since we only fetch and summarize.

We will need the language model (LM) for only one step during the process, which is taking in HTML data from an article web page and extracting the article's content and some data about it. Feeding the model with the HTML data will be achieved by using deterministic requests with Python, not being in the hands of the LM as this clears out a big surface of unexpected behavior - well, at least in the first (later nested) step.

We will use LangGraph to create a custom Agent (which is not very agentic yet, but can be reused more straightforwardly). 

But before we do that, we will anticipate defining the structured output, that we will require the model to output when feeding it HTML data. Luckily, Langchain lets us use Pydantic Models, which allow us to define forms with fields that will basically be filled out by the LM:

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class SummarySchema(BaseModel):

    #Fields
    title: str = Field(description="Title of the news article")
    url: str = Field(description="URL that was used to load the news article")
    summary: str = Field(description="Summary of the news article")
    content: str = Field(description="Full content of the news article")
    category: Literal["AI", "Hardware", "Software Development", "Web Development", "Cloud Computing", "Data Science", "Machine Learning", "Cybersecurity", "Networking", "DevOps", "Mobile Development", "Game Development", "Database Management", "IT", "Project Management"]
    datePublished: str = Field(description="Date when the news article was published, format YYYY-MM-DD")
    imageUrl: Optional[str] = Field(default=None, description="URL of the image associated with the news article")
    source: str = Field(description="Publishing source of the news article")

    # Method that ensures custom constraints on a certain field
    @field_validator("datePublished")
    def validate_date(cls, v):
        try:
            date = datetime.fromisoformat(v)
            return datetime.strftime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("datePublished is not a valid date")
```

this is seriously powerful because we do not have to parse any keywords as attributes and rely on syntax consistencies and actually receive an instance of this class with all its' attributes set by the LM. 

### [Agentic Unit](#model-based-article-summarization)
<a id="agentic-unit"></a>
We can then feed this class to bind it to the model that we use in the agent. To build an agent, we use a builder instance that is given a type of state. State objects hold runtime working data being passed along the workflow. They are classes with attributes that can be accessed as needed by a node. 

A **node** is a unit of work that does interactions and prompts ai models - they are simply functions that take a state and return an object with the attribute(s) of the state object to be updated.

The simplest form of **State** objects are **MessagesState objects**. They only have an attribute "messages", which is a **list of Message objects** and are semi-intuitively managing the runtime output data as a chat history.

States are passed through the agent along nodes and their connecting edges. **Edges** between nodes are used to design conditional or chained agent processes. We only have one edge that is the minimum requirement to get your agent to actually start doing something. In combination with structured outputs and programmatic conditional statements, we could effectively orchestrate more complex decisions.

```python
from langgraph.graph import StateGraph, START, MessagesState
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage

def get_article_extractor(model: ChatOllama):
    # Bind structured output
    model = model.with_structured_output(SummarySchema)

    # Instantiate new Agent builder
    builder = StateGraph(MessagesState)
    
    # Define a Node
    async def summarize(state: MessagesState):
        print("CHECKPOINT Summarize Node")
        # The chain of messages will be stored in the state object. We just assume that the HTML data is in the latest message of this chain.
        html = state["messages"][-1]

        # We just define a usual model invocation as our ai piece of work
        response = await model.ainvoke(
            [
                SystemMessage(content=f"You are a web page content summarizer. Use the given output structure to summarize the page content."),
                SystemMessage(content=html.content)
            ]
        )
        # The response variable will be where our instantiated Pytantic object will be stored. 
        # We manually add the contents as JSON data to our node output with Pydantic's built-in json dumper.
        # Wayyy more robust than instructing the LM to respect a format
        return { "messages":  AIMessage(content=response.model_dump_json(indent=2))}
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "summarize")
    # CompiledGraph can be prompted like a language model
    return builder.compile()
```

## [Prepare Agents with different Models](#model-based-article-summarization)
<a id="prepare-agents-with-different-models"></a>
We now have our 'agentic' workflow unit, that we can invoke directly. Lets now prepare our different configurations, we want to test against each other.
Use this cell to define your choice of Agents to compare. You can simply comment out lines of configurations you want to exclude, but not delete.

Later, the batch runner will iterate over these agents and feed them with the same data

```python
from typing import Dict
from langgraph.graph.graph import CompiledGraph

AGENTS: Dict[str, CompiledGraph] = {
    "qwen3:8b": get_article_extractor(get_model("qwen3:8b")),
    #"qwen3:8b-0DEG": get_article_extractor(get_model("qwen3:8b", temperature=0)),
    "granite3.2": get_article_extractor(get_model("granite3.2")),
    #"qwq": get_article_extractor(get_model("qwq"))
}
```

## [Runner](#model-based-article-summarization)
<a id="runner"></a>
After setting up our Agent arsenal, which all convert HTML text into structured output, we now need logic to take a URL, load the contents and then feed these to the agents to receive their summaries.
We will build two runner types:
1. a batch runner, that loads the article once and runs it through different agents
2. a unit runner, that can be exposed as a tool for another agent or chat bot to use per URL

### [HTML Retrieval](#model-based-article-summarization)
<a id="html-retrieval"></a>
We have not talked yet about what to feed the models. Since, we do not want them to take care of calling a fetching tool to load the URLs website data, we will just prepend this step programatically.

We are using the simplest of the ways to retrieve a URLs response and wrap it into a big string with the original URL attached, so the LM knows where it came from.

```python
import requests

# Function that returns the HTML for a URL
def page2string(url: str) -> str:
    page_response = requests.get(url)
    html = page_response.content
    return f"Page content for '{url}':\n{html}"
```

Some other important thing is, that programmatically loading websites, especially monetized content, is, sometimes adversing their anti-robot measures. While this workflow is tested, you should be fine with using real articles. The batch runner loads the article once, and then passes the content to each agent.

However, if you expect to repeatedly call the same URLs for playing around, you should consider using a mock website, using Python's standard HTTP library. Simply load an article in your browser and then save it locally as HTML. In a terminal, in the directory of this HTML file, run:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

this will make python serve your current directory via HTTP. You can then just access ``http://localhost:8000/<some-page>.html`` in your runner.

### [Batch Runner](#model-based-article-summarization)
<a id="batch-runner"></a>
Now, we arrived at the top-level cell that can be repeatedly run directly to actually do the work. Everything before that was just definitions and compiling (when making our AGENTS variable). If you change anything in the cells above, you have to also re-construct the AGENTS variable - might as well just re-run everything "up to this cell".

```python
from langchain_core.messages import HumanMessage

url = "http://localhost:8000/page.html"

# Load url content and convert to a Message object
html = page2string(url)
prompt = HumanMessage(content=html)

# actual runner
async def stream_and_pretty_print(agent: CompiledGraph, prompt: HumanMessage):
    message_index = 1 # skip showing the raw HTML
    async for chunk in agent.astream({"messages": [prompt]}, stream_mode="values"):
        new_messages = chunk["messages"][message_index:]
        message_index += len(new_messages)
        for message in new_messages:
            message.pretty_print()

# per agent configuration
for model, agent in AGENTS.items():
    # identify configuration in output
    print(f"STARTING EXTRACTION WITH '{model}' AGENT")
    # run runner
    await stream_and_pretty_print(agent, prompt)
```

### [Unit Runner (Tool)](#model-based-article-summarization)
<a id="unit-runner-tool"></a>
Wrapping slightly differently - not stream-printing, but actually returning the ouput and using a fixed agent configuration (you might need to adjust, based on your ollama models available).

This runner makes our agent usable as a tool in other agents and models, if you chose to spin up your own below here.

```python
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# decorate any python function that returns a string with this, to turn it into a tool object
@tool
async def summarize_article(url: str):
    # the docstring is important for the LM to understand this tool
    """
    This tool takes a url, expecting an article page, as input, loads it and returns a structured summary based on the URL's page data.
    """
    html = page2string(url)
    prompt = HumanMessage(content=html)

    response = await AGENTS["granite3.2"].ainvoke({"messages": [prompt]})
    return response["messages"][-1].content # tool answer will be only the final output (nothing from history)
```
