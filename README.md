# aigency
repo for learning AI agents and MCP

## Demos
Starting to explore agents through different demos.

### [Claude PowerPoint Assistant](./claude-powerpoint-assistant/README.md)
Using Claude Desktop, we setup an MCP-Server and hook it to Claude's MCP-Client configuration. This third-party MCP-Server will enable an LM to interact with running PowerPoint applications.

### [AI-induced Article Summarizer Worker](./langchain-summarizer/README.md)
Based on Ollama and the LangChain ecosystem, we build a simple workflow that takes a URL in the beginning, downloads the HTML and has an LM extract a fixed-format summary about the content's of the web page - all running locally.