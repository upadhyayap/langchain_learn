from langchain_community.tools.tavily_search import TavilySearchResults

def get_profile_url_tavily(name: str) -> str:
    """Search the web to find a LinkedIn profile url"""
    search = TavilySearchResults(max_results=10)
    return search.run(f"{name} linkedin profile")




