from langchain_core.tools import tool


class GmailTool:
    def __init__(self, model):
        self.model = model

    @tool
    def send_email(self):
        pass