from langchain_community.tools import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults

class SearchTool:
    # search = TavilySearchResults(
    #     max_results=2,
    #     search_depth="advanced",
    #     include_answer=True,
    #     include_raw_content=True,
    #     include_images=True
    # )

    search = DuckDuckGoSearchResults()