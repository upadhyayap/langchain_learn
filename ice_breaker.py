import os
from typing import Tuple
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from azure.core.credentials import AzureKeyCredential
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import Summary, summary_parser

def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url, mock=True)
    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    \n{format_instructions}
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
    )

    llm_azure_openai = AzureChatOpenAI(
        temperature=0,
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model=os.environ["AZURE_OPENAI_MODEL"],
    )
    llm_ollama = ChatOllama(model="llama3.2")

    chain_azure_openai = summary_prompt_template | llm_azure_openai | summary_parser
    chain_ollama = summary_prompt_template | llm_ollama | StrOutputParser()

    res: Summary = chain_azure_openai.invoke(input={"information": linkedin_data})
    # print("Azure OpenAI response")
    # print(res)

    # res_ollama = chain_ollama.invoke(input={"information": linkedin_data})
    # print("Ollama response")
    # print(res_ollama)
    
    return res, linkedin_data.get("photoUrl")

if __name__ == "__main__":
    load_dotenv()
    ice_break_with("Anand upadhyay working in endorlabs")
