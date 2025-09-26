import os
import gradio as gr
import requests
import inspect
import pandas as pd

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_react_agent, AgentExecutor
from langfuse.langchain import CallbackHandler
from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()


from langchain.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs

@tool
def get_youtube_transcript(url: str) -> str:
    """
    Returns the transcript of a YouTube video. 
    If transcripts are disabled, unavailable, or if the IP is blocked by YouTube,
    it will return a clear error message.
    """
    def get_video_id(url_str: str):
        # (Helper function remains the same)
        query = urlparse(url_str)
        if query.hostname == 'youtu.be': return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch': return parse_qs(query.query).get('v', [None])[0]
            if query.path[:7] == '/embed/': return query.path.split('/')[2]
            if query.path[:3] == '/v/': return query.path.split('/')[2]
        return None

    video_id = get_video_id(url)
    if not video_id:
        return "Error: Invalid YouTube URL format. Could not extract video ID."

    try:
        api_instance = YouTubeTranscriptApi()
        transcript_list = api_instance.list(video_id)
        transcript = transcript_list.find_transcript(['en', 'en-US'])
        transcript_data = transcript.fetch()
        transcript_text = " ".join([entry.text for entry in transcript_data])
        return transcript_text if transcript_text else "Error: Transcript was found but is empty."
    
    except TranscriptsDisabled:
        return "Error: Transcripts are officially disabled for this video."
    except NoTranscriptFound:
        return "Error: Could not find an English transcript for this video."
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"


###########################################################################################################
    
import json
import re
from langchain.agents.output_parsers.react_json_single_input import ReActJsonSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish

class UniversalReactOutputParser(ReActJsonSingleInputOutputParser):
    """
    A robust, universal output parser that handles both JSON and plain string inputs.
    It cleans model-specific tags and can gracefully handle inconsistencies in LLM output
    for different tools.
    """
    def parse(self, text: str) -> AgentAction | AgentFinish:
        # 1. Clean any model-specific tags from the text
        text = text.replace("<think>", "").replace("</think>", "").strip()

        # 2. Check for the final answer and finish the run if found
        if "FINAL ANSWER:" in text:
            return AgentFinish(
                {"output": text.split("FINAL ANSWER:")[-1].strip()}, text
            )

        # 3. Primary Strategy: Try to parse as JSON (for Tavily, etc.)
        try:
            # Attempt to use the parent class's JSON parsing logic
            response = super().parse(text)
            return response
        except ValueError:
            # 4. Fallback Strategy: If JSON parsing fails, treat as plain text
            print("--- JSON parsing failed. Falling back to plain text parsing. ---")
            
            # Use regex to find the action and the entire action input block
            action_match = re.search(r"Action:\s*(.*?)\s*Action Input:\s*(.*)", text, re.DOTALL)
            if not action_match:
                # If the regex fails, it's an unrecoverable format error
                raise ValueError(f"Could not parse LLM output: {text}")

            action = action_match.group(1).strip()
            action_input = action_match.group(2).strip(" '\"") # Clean wrapping quotes

            return AgentAction(action, action_input, text)
###############################################################################################
output_parser = UniversalReactOutputParser() 

def create_agent(model_source="qwen", api_key=None):
    """
    This function creates and returns a LangChain agent executor.
    An agent executor is the runtime for an agent. It's what calls the agent, executes the tools, 
    and passes the tool outputs back to the agent to figure out the next step.
    """
    print("Initializing LangChain agent...")

    # 1. Select the LLM based on the user's choice
    if model_source == "Hugging Face":
        # if not api_key:
        #     raise ValueError("Hugging Face API Key is required.")
        # os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            task="text-generation", max_new_tokens=512, temperature=0.1
        )
        print(f"LLM selected: {model_source}")
    elif model_source == "Gemini":
        # if not api_key:
        #     raise ValueError("Google API Key is required.")
        # os.environ["GOOGLE_API_KEY"] = api_key
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        print(f"LLM selected: {model_source}")
    elif model_source == "qwen": 
        llm = ChatOllama(model="qwen3:8b", temperature=0)
        print(f"LLM selected: {model_source}")
    else: # Default to Ollama
        llm = ChatOllama(model="llama3", temperature=0)
        print(f"LLM selected: {model_source}")


    # 2. Define the Tools
    # These are the capabilities the agent can use.
    # GAIA requires web search and code execution.
    search_tool = TavilySearchResults(max_results=3)
    python_repl_tool = PythonREPLTool()
    
    tools = [search_tool, python_repl_tool, get_youtube_transcript]

    # 3. Create the Custom Prompt Template 
    # We are now defining a multi-line string that contains the full prompt,
    # incorporating the GAIA formatting rules directly.
    prompt_template = """You are a helpful AI assistant for GAIA evaluation. Your goal is to answer the user's question accurately. You have access to tools to help you.

    **TOOLS:**
    Here are the tools you can use:
    {tools}

    **INSTRUCTIONS ON HOW TO RESPOND:**
    You must follow the ReAct (Reasoning and Acting) format. Your response should be a sequence of Thought/Action/Action Input/Observation steps. At the end of your reasoning, you MUST provide a final answer.

    **INSTRUCTION: HOW TO HANDLE TOOL ERRORS**
    If a tool returns an error, your next "Thought" must be to analyze the error and try a different strategy. Your primary backup strategy is to use the 'tavily_search_results_json' tool to find an alternative way to answer the question.

    **Use this format:**

    Question: The input question you must answer
    Thought: Your reasoning about what to do next. Analyze the question and decide if you need a tool.
    Action: The name of the tool to use, which must be one of [{tool_names}]
    Action Input: The input for the tool, in a JSON dictionary format.
    Observation: The result returned by the tool.
    ... (this Thought/Action/Action Input/Observation cycle can repeat multiple times) ...
    Thought: I now have all the information I need to provide the final answer.
    FINAL ANSWER: [YOUR FINAL ANSWER]

    ---
    **RULES FOR THE FINAL ANSWER:**
    The text following "FINAL ANSWER:" must follow these strict rules:

    1.  **Format**: Your final answer must be a number, as few words as possible, or a comma-separated list.
    2.  **Numbers**: Do not use commas (e.g., `12345`) or units (e.g., `15`, not `$15` or `15%`) unless the question specifically asks for it.
    3.  **Strings**: Do not use articles (e.g., `Eiffel Tower`, not `the Eiffel Tower`). Do not use abbreviations. Write numbers as words (e.g., `four`, not `4`).
    4.  **Lists**: Apply the number and string rules to each element of the list.

    **Begin!**

    Question: {input}
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(prompt_template)
    print("Custom GAIA prompt created.")

    # 4. Create the Agent
    # This binds the LLM, tools, and prompt together into a runnable agent.
    agent = create_react_agent(llm, tools, prompt, output_parser=output_parser)
    print("LangChain agent created.")

    # 5. Create the Agent Executor
    # This is the final object that we will call to run the agent logic.
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, # Set to True to see the agent's thoughts
        handle_parsing_errors=True, 
        max_iterations=4,
        early_stopping_method="force"  # Force answer on limit
    )
    print("Agent Executor created.")
    
    return agent_executor



# This is a wrapper class to make the agent executor compatible with the existing script structure.
class LangChainAgent:
    def __init__(self):
        self.agent_executor = create_agent()
        # Initialize the Langfuse handler. It will automatically use the environment variables.
        # We give it a session_id to group all runs within a single "Run Evaluation" click.
        self.langfuse_handler = CallbackHandler()
        print("Langfuse CallbackHandler initialized.")

    def __call__(self, question: str, task_id: str) -> str:
        print(f"Agent received question: {question[:100]}...")
        try:

             # We will pass the task_id as metadata to easily find this specific run in Langfuse.
            config = {
                "callbacks": [self.langfuse_handler],
                "metadata": {
                    "task_id": task_id,
                }
            }

            # The agent executor returns a dictionary, the final answer is in the 'output' key.
            response = self.agent_executor.invoke({"input": question}, config=config)
            print("START________________________________________________________________________")
            print(response)
            print("END_____________________________________________")
            answer = response.get("output", "Agent failed to produce an answer.")
            print(f"Agent returned answer: {answer}")
            return str(answer)
        except Exception as e:
            print(f"An error occurred while running the agent: {e}")
            return f"Error: {e}"


def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    username = "Azazzz11" 

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = LangChainAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    questions_data = questions_data[:2]
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text, task_id=task_id)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    # gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)