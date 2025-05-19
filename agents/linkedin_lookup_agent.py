import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool, Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from tools.tools import get_profile_url_tavily

load_dotenv()

def lookup(name: str) -> str:
    llm = AzureChatOpenAI(
        temperature=0,
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model=os.environ["AZURE_OPENAI_MODEL"],
    )
    template = """
    Given the name {name} I want you to get the LinkedIn url of the person.your output should be a valid url.
    """
    prompt_template = PromptTemplate(
        input_variables=["name"],
        template=template
    )
    tools_for_agent = [
        Tool(
            name="crawl google 4 linkedin profile",
            description="useful when you need to get the LinkedIn url of a person",
            func=get_profile_url_tavily
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools_for_agent, react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)
    res =  agent_executor.invoke(
        {
            "input": prompt_template.format_prompt(name=name)
        }
    )

    return res["output"]
    


if __name__ == "__main__":
    linkedin_profile_url = lookup("Anand upadhyay working in endorlabs")
    print(linkedin_profile_url)



