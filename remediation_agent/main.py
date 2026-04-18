import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

load_dotenv()

def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        temperature=0,
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=SecretStr(os.environ["AZURE_OPENAI_API_KEY"]),
        model=os.environ["AZURE_OPENAI_MODEL"],
    )

system_prompt = """
"System"
You are a Docker security expert that analyzes Dockerfiles and fixes security vulnerabilities while preserving the original base image and core functionality.

you will be given a Dockerfile and a package version json file that has a list of package versions that are found in container image at the key "spec.resolved_dependencies.dependencies" and a dependency graph of the packages can be found at the key "spec.resolved_dependencies.dependency_graph".

you will need to fix the Dockerfile to make it more secure.

you will need to return the Dockerfile with the fixes.

Fix the Dockerfile to make as secure as possible. Think hard before suggesting any changes.

output format:
{{
    A valid Dockerfile with the fixes.
}}
"""

def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()

def main():
    chart_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Dockerfile: \n\n {dockerfile} \n\n Packages version json: \n\n {package_version} \n\n"),
    ])
    
    llm_azure_openai = get_llm()

    chain = chart_prompt | llm_azure_openai

    res = chain.invoke({
        "dockerfile": read_file("/Users/anand.upadhayay/endordev/container_scan/spikes/Dockerfile"),
        "package_version": read_file("/Users/anand.upadhayay/endordev/container_scan/spikes/container_pv.json"),
    })
    print(res.content)


if __name__ == "__main__":
    main()