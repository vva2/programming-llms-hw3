from composio_langchain import ComposioToolSet, Action

class CalendarTools:
    tools = ComposioToolSet().get_tools(actions=[
        Action.GOOGLECALENDAR_CREATE_EVENT,
        Action.GOOGLECALENDAR_DELETE_EVENT,
        Action.GOOGLECALENDAR_FIND_FREE_SLOTS,
        Action.GOOGLECALENDAR_UPDATE_EVENT,
        Action.GOOGLECALENDAR_FIND_EVENT,
        Action.GOOGLECALENDAR_QUICK_ADD,
        Action.GOOGLECALENDAR_GET_CURRENT_DATE_TIME
    ])