from langchain_core.tools import tool
from pydantic import BaseModel, Field

from models.models import ModelFactory
from utils.loggerr import logger
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
import base64
from email.mime.text import MIMEText
import os
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials


class Email(BaseModel):
    """Email model"""
    recipient_email: str = Field(description="recipient email address properly worded. Fix grammatical errors if any.")
    subject: str = Field(description="subject of the email properly worded. Fix grammatical errors if any.")
    body: str = Field(description="body of the email including the sender signature. everything properly worded. Fix grammatical errors if any.")

class GmailTool:
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']

    def get_missing_fields(draft: Email):
        missing_fields = []

        if draft.subject == '<UNKNOWN>':
            missing_fields.append('subject')
        if draft.body == '<UNKNOWN>':
            missing_fields.append('body')
        if draft.recipient_email == '<UNKNOWN>':
            missing_fields.append('recipient email')

        return missing_fields

    def send_email_using_gmail_api(draft: Email):
        # send email using api
        creds = None
        if os.path.exists(os.getenv('GMAIL_API_TOKEN_FILE')):
            creds = Credentials.from_authorized_user_file(os.getenv('GMAIL_API_TOKEN_FILE'), GmailTool.SCOPES)
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(os.getenv('GMAIL_ACCOUNT_SECRET_FILE'), GmailTool.SCOPES)
            creds = flow.run_local_server(port=0)
            with open(os.getenv('GMAIL_API_TOKEN_FILE'), 'w') as token:
                token.write(creds.to_json())

        service = build('gmail', 'v1', credentials=creds)

        message = MIMEText(draft.body)
        message['to'] = draft.recipient_email
        message['subject'] = draft.subject
        create_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

        return service.users().messages().send(userId="me", body=create_message).execute()

    @tool
    def send_email(recipient_email: str, message: str):
        """sends an email. call this function only when recipient email, and message are explicitly provided by the user and the recipient_email is a valid email."""
        draft = (ModelFactory
                 .public_model
                 .with_structured_output(Email)
                 .invoke(f"recipient={recipient_email}, message={message}"))

        missing_fields = GmailTool.get_missing_fields(draft)

        if missing_fields.__len__() > 0:
            return f"missing information: {missing_fields}. Please confirm additional info from the user. Here are the details of the draft: {draft.__dict__}"

        print(f"Sending the email with the following info:\n{draft.__dict__}\n")

        user_input = input("Would you like to proceed? (y/n) ")

        if user_input.strip().lower() != 'y':
            return f"Sender (User) denied the send email request. May not be pleased with the draft. Ask the user the reason for denying the request."

        logger.info('inside send_email')
        logger.info(f'recipient email: {draft.recipient_email}')
        logger.info(f'subject: {draft.subject}')
        logger.info(f'body: {draft.body}')

        try:
            message = GmailTool.send_email_using_gmail_api(draft)
            logger.info(F'Message Id: {message["id"]}')
        except Exception as error:
            logger.error(f'An error occurred: {error}')
            return f'An error occurred: {error}'

        return "email sent!"