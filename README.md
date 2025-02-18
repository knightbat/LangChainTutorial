# LangChain Project

This project is a question-answering application that uses LangChain to retrieve and process information from URLs. The application extracts data from the provided URLs, splits the data into chunks, and uses a language model to answer questions based on the retrieved information.

## Features

- Extracts data from URLs
- Splits data into manageable chunks
- Uses FAISS for vector storage and retrieval
- Utilizes a language model for question answering
- Provides sources for the answers

## Requirements

- Python 3.12
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/knightbat/LangChainTutorial.git
    cd LangChainTutorial
    ```

2. Create a Conda environment and activate it:
    ```sh
    conda create --name langchain-env python=3.12
    conda activate langchain-env
    ```

3. Install the required packages:
    ```sh
    conda install --file requirements.txt
    ```

## Usage

1. Run the application:
    ```sh
    streamlit run main.py
    ```

2. Open the application in your web browser. You will see a sidebar where you can input up to three URLs.

3. Click the "Submit" button to load data from the URLs.

4. Enter your question in the provided text input field and press Enter.

5. The application will display the answer along with the sources.

## Code Overview

### `main.py`

- **data_extraction**: Extracts data from the provided URLs, splits the data into chunks, and saves the index.
- **get_response**: Loads the index, creates a custom prompt, and retrieves the answer to the query using the language model.
- **main**: The main function that sets up the Streamlit interface, handles user input, and displays the results.

Reference:
    For more information, you can watch this video: https://www.youtube.com/watch?v=d4yCWBGFCEs
"""