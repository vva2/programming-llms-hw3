from langchain_community.tools import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults

class SearchTools:
    tools = [DuckDuckGoSearchResults()]