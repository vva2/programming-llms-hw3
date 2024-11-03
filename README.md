# 3-HW

# Installation Guide

This guide will help you set up the personal assistant powered by LangChain and LangGraph.

## Prerequisites

Before starting the installation, make sure you have Python (I used Python 3.12.3) installed on your system.

## Installation Steps

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Ollama**
    - Visit [Ollama's official website](https://ollama.ai) to download and install Ollama for your operating system
    - After installation, pull the required model:
      ```bash
      ollama pull mistral-nemo
      ```

3. **Set Up Environment Variables**
   Create a `.env` file in the root directory of the project with the following content:
   ```env
   ANTHROPIC_API_KEY='[FILL HERE]'
   ANTHROPIC_3_5_MODEL_V3='claude-3-opus-20240229'
   ANTHROPIC_3_5_MODEL_V2='claude-3-5-sonnet-20241022'
   ANTHROPIC_3_5_MODEL_V1='claude-3-5-sonnet-20240620'
   MISTRAL_NEMO_12B_MODEL='mistral-nemo'
   COMPOSIO_API_KEY='[FILL HERE]'
   COMPOSIO_LOGGING_LEVEL='ERROR'
   CHROMA_DB_PATH='chroma.db'
   FULLY_LOCAL=0
   SKIP_PRIVACY_CHECK=0
   CONTEXT_HISTORY_LEN=20
   ANTHROPIC_3_MODEL='claude-3-sonnet-20240229'
   ```

4. **Obtain API Keys**
    - **Anthropic API Key**:
        - Sign up at [Anthropic's website](https://www.anthropic.com)
        - Navigate to your account settings to generate an API key
        - Copy the key and paste it in the `.env` file

    - **Composio API Key**:
        - Visit [Composio Settings](https://app.composio.dev/settings)
        - Generate your API key
        - Copy the key and paste it in the `.env` file

5. **Configure Composio Integrations**
   After obtaining your Composio API key, set up the required integrations:
   ```bash
   composio add googlecalendar
   composio add gmail
   ```
   Follow the prompts to complete the authentication process for both services.

## Running the Application

To run the application, navigate to the root of the project and execute:

```bash
python script.py
```

## Verification

To verify your installation:

1. Ensure all environment variables are properly set
2. Confirm Ollama is running and the Mistral model is downloaded
3. Verify that Composio integrations are properly connected

## Troubleshooting

If you encounter any issues:

- Check that all API keys are correctly entered in the `.env` file
- Ensure Ollama is running before attempting to use the assistant
- Verify that your Python environment matches the requirements
- Check that Composio integrations were successfully authenticated

--- 

This guide now includes instructions on how to run the application. Let me know if you need any more details!