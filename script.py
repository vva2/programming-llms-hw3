from dotenv import load_dotenv

load_dotenv()

import os
import threading
import itertools
import time
from Agent import Agent
from langchain_core.messages import HumanMessage
from models.models import get_local_model
from loggerr import logger
import click
import readline

class Assistant:
    def __init__(self):
        self.agent = Agent().agent
        self.config = {"configurable": {"thread_id": 53}}
        self.loading = False
        self._spinner = None

    def _spinner_animation(self):
        spinner = itertools.cycle(['|', '/', '-', '\\'])
        while self.loading:
            print(f"\rAssistant is thinking... {next(spinner)}", end="", flush=True)
            time.sleep(0.1)
        print("\r" + " " * 50, end="", flush=True)
        print("\r", end="", flush=True)

    def can_proceed_safely(self, user_input: str, local_model) -> bool:
        logger.info(f"Checking '{user_input}' for privacy violations.")

        private_data_check_response = local_model.invoke(
            [
                HumanMessage(
                    content=f"Identify the private information in this user prompt. If the private information is not explicitly mentioned then simply dont report anything.:\n-----------`{user_input}`\n-----------"
                )
            ]
        )

        prompt = f"The following response describes the private data found in an user prompt. Using this data answer 'Yes' (if private data is indeed present) or 'No' (otherwise or if you are unsure). BE CONCISE I DO NOT NEED ANY EXPLANATION.\n-----\n{private_data_check_response.content}\n-----"

        result = get_local_model().invoke([HumanMessage(prompt)])

        logger.info(f'response: {result} for prompt: {prompt}')

        if result and 'yes' in result.content.lower():
            self._stop_spinner()
            click.echo(click.style(f"\n!!PRIVATE INFO FOUND!!\n{private_data_check_response.content}\n", fg="yellow"))
            user_consent = click.prompt("\nDo you want to proceed sending this info to a public LLM? (y/n): ")

            if user_consent.lower().strip() != 'y':
                click.echo(
                    click.style(f"Your most recent input has been cleared from memory. Please re-phrase it for me.",
                                fg="yellow"))
                return False

        return True

    def _stop_spinner(self):
        # Stop the spinner
        self.loading = False

        if self._spinner:
            self._spinner.join()
            self._spinner = None

    # Function to simulate the assistant's response
    def assistant_response(self, user_input) -> str:
        self._spinner = None
        try:
            # Start the spinner in a separate thread
            self.loading = True
            self._spinner = threading.Thread(target=self._spinner_animation)
            self._spinner.start()

            if len(user_input) == 0:
                self._stop_spinner()
                return "Empty input received. Please try again!"

            if int(os.getenv('FULLY_LOCAL')) == 0 and int(
                    os.getenv('SKIP_PRIVACY_CHECK')) == 0 and not self.can_proceed_safely(
                    user_input, get_local_model()):
                self._stop_spinner()
                return "Input erased from memory. Let's try again."

            logger.info(f'User input: {user_input}')

            response = self.agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=self.config)

            self._stop_spinner()

            logger.info(response)
            return response['messages'][-1].content
        except Exception as e:
            self._stop_spinner()

            logger.error(f'Error Occurred: {e}')
            return "Error Occurred. Please try again."
        finally:
            # Ensure the spinner stops even if an exception occurs
            self.loading = False

    # Function to read multi-line input with history support
    def get_multiline_input_with_history(self, prompt):
        click.echo(click.style(prompt + " (end with an empty line):", fg="green"))
        lines = []

        while True:
            line = input()
            if line == "":
                break
            lines.append(line)

        # Add the complete input to the history
        if lines:
            readline.add_history("\n".join(lines))

        return "\n".join(lines)

    def respond_assistant(self, response):
        click.echo(f"{click.style("Assistant>", fg="red")}\n{response}\n")

    # Main function to handle the chat
    def chat(self):
        click.echo("Welcome to the CLI Chat!")
        click.echo("Type 'exit' or '\\q' to end and '\\c' to clear the chat.\n")

        while True:
            try:
                # Get multi-line input with history support
                user_input = self.get_multiline_input_with_history("User")

                # Check if the user wants to exit
                if user_input.lower() == "exit" or user_input.lower() == "\\q":
                    click.echo(click.style("\nExiting the chat. Goodbye!", fg="yellow"))
                    break

                if user_input.lower() == "\\c":
                    click.clear()
                    continue

                # Generate and display the assistant's response
                response = self.assistant_response(user_input)
                self.respond_assistant(response)
            except KeyboardInterrupt:
                click.echo(click.style("\nExiting the chat. Goodbye!", fg="yellow"))
                break


def clear_screen_event():
    logger.info("Ctrl + L detected! Clearing the screen...")
    click.clear()

@click.command()
def main():
    assistant = Assistant()
    assistant.chat()

if __name__ == "__main__":
    main()
