# DBChat: Chat with your Database

A Streamlit application that allows users to query databases using natural language.

## Features

- Connect to PostgreSQL, MySQL, or SQLite databases
- Chat with your database using natural language
- Powered by LangChain and Groq AI models
- Streamlit-based user interface

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your environment variables in a `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

## Usage

1. Select your database type from the sidebar
2. Enter your database connection details
3. Click "Connect to Database"
4. Start chatting with your database using natural language!

## Requirements

See `requirements.txt` for a full list of dependencies.

## License

MIT

