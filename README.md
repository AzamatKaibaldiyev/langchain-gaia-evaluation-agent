# LangChain Agent for GAIA Evaluation

This repository contains a sophisticated AI agent designed to compete in the **GAIA (General AI Assistants)** benchmark, a challenging evaluation for language agents. The project was developed as the final assignment for the Hugging Face Agents Course.

The agent leverages the **LangChain** framework and is built with a robust, multi-tool architecture. It features a custom universal parser to handle diverse model outputs and is capable of using web search, code execution, and video transcript analysis to answer a wide range of questions.


---

## 🚀 Features

* **Multi-Tool Capability**: Integrates multiple tools to solve complex problems:
    * `TavilySearchResults`: For fast and accurate web searches.
    * `PythonREPLTool`: For executing Python code to perform calculations or string manipulations.
    * `get_youtube_transcript`: A custom tool to fetch and analyze YouTube video transcripts.
* **Universal ReAct Parser**: A custom-built output parser (`UniversalReactOutputParser`) that intelligently handles both JSON and plain-text tool inputs, making the agent compatible with a wide range of LLMs.
* **Resilient Error Handling**: The agent's prompt and tools are designed to gracefully handle failures (e.g., unavailable video transcripts, IP blocks), allowing it to pivot its strategy instead of terminating.
* **Model Agnostic**: Easily configurable to run with various LLMs, including local models via Ollama (like `qwen` or `llama3`) and API-based models from Google (Gemini).
* **Gradio Interface**: Includes a simple Gradio web UI to run the full GAIA evaluation and submit the agent's answers for scoring.

---

## 🛠️ Tech Stack

* **Framework**: [LangChain](https://www.langchain.com/)
* **LLMs**: Configured for [Ollama](https://ollama.com/), [Google Gemini](https://ai.google.dev/)
* **Core Tools**: [Tavily Search API](https://tavily.com/), Python REPL
* **UI**: [Gradio](https://www.gradio.app/)
* **Deployment**: Hugging Face Spaces

---

## ⚙️ Setup and Installation

Follow these steps to set up and run the agent on your local machine.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate
# On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a file named `.env` in the root of the project directory. This file will store your API keys.

```env
# .env file

# Required for the Tavily Search tool
TAVILY_API_KEY="tvly-..."

# Required if using the Gemini model
GOOGLE_API_KEY="AIza..."
```

### 5. Run the Application

Launch the Gradio interface by running the `app.py` script.

```bash
python app.py
```

You can now access the web interface in your browser (usually at `http://127.0.0.1:7860`) to run the evaluation.

---

## 🏛️ Agent Architecture

The agent's design is centered around the **ReAct (Reasoning and Acting)** framework, which enables it to reason about a problem and choose the appropriate tool to solve it.

1.  **Prompt Template**: The core logic is guided by a detailed prompt that instructs the agent on how to use tools, handle errors, and format its final answer according to GAIA's strict rules.
2.  **LLM**: The "brain" of the agent. The code is flexible enough to use different models, with the default set to a local `qwen` model via Ollama for privacy and cost-effectiveness.
3.  **Tools**: The agent's "hands." It has a set of capabilities that it can invoke to interact with the outside world, find information, or perform computations.
4.  **Universal Parser**: This custom component is the "ears." It listens to the LLM's output and reliably translates its intent into a structured `AgentAction` or a `AgentFinish` signal, even when the LLM's output format is inconsistent.
5.  **Agent Executor**: The main runtime that orchestrates the loop: it sends the prompt to the LLM, receives the desired action, calls the corresponding tool, gets the result, and feeds it back to the LLM until the problem is solved.