from composio_langchain import ComposioToolSet, Action

class GmailTools:
    tools = ComposioToolSet().get_tools(actions=[
        Action.GMAIL_SEND_EMAIL,
        Action.GMAIL_CREATE_EMAIL_DRAFT,
        Action.GMAIL_GET_PROFILE
    ])