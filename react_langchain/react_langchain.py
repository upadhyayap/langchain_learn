import os
from typing import Union
from dotenv import load_dotenv
from langchain.agents import tool
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.tools.render import render_text_description
from langchain_openai import AzureChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.format_scratchpad import format_log_to_str
from callbacks import AgentCallbackHandler

load_dotenv()

def find_tools_by_name(tools: list[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

@tool
def get_txt_length(txt: str) -> int:
    """
    Returns the length of a text by characters
    """
    print(f"entering get_txt_length: {txt}")
    txt = txt.strip("'\n").strip(
        '"'
    )
    return len(txt)


if __name__ == "__main__":
    print("Hello from react-langchain!")
    tools = [get_txt_length]
    tool_names = [t.name for t in tools]

    template="""Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]))

    llm_azure_openai = AzureChatOpenAI(
        temperature=0,
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model=os.environ["AZURE_OPENAI_MODEL"],
        stop=["\nObservation"],
        callbacks=[AgentCallbackHandler()],
    )
    intermediate_steps = []

    agent = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
    } | prompt | llm_azure_openai | ReActSingleInputOutputParser()

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({
        "input": "What is the length of the text Hello, world!?",
        "agent_scratchpad": intermediate_steps
    })
    print(f"Agent Step: {agent_step}")

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tools_to_use = find_tools_by_name(tools, tool_name)

        tool_input = agent_step.tool_input
        observation = tools_to_use.func(str(tool_input))
        print(f"Observation: {observation}")
        intermediate_steps.append((agent_step, observation))
    if isinstance(agent_step, AgentFinish):
        print(f"Final Answer: {agent_step.return_values}")
    