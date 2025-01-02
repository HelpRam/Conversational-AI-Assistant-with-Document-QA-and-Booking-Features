# Conversational-AI-Assistant-with-Document-QA-and-Booking-Features

## Description

This project implements a **Chatbot** that can answer user queries based on the content of uploaded documents (e.g., PDFs) and also includes a **conversational form** for collecting user information (Name, Phone Number, Email) when the user requests a callback. Additionally, the chatbot allows users to **book appointments** by selecting dates and times, with automatic **date extraction** and **input validation** for phone numbers and emails.

The project leverages **LangChain**, **Google Gemini**  and other libraries to fulfill the requirements of the task.

## Features

- **Document Querying**: Upload a PDF and ask questions related to its content. The chatbot will use **LangChain** to retrieve the relevant information from the document.
- **Callback Request Form**: Collects user information (Name, Phone Number, Email) with validations:
  - **Phone Number**: Validates the phone number as a 10-digit number.
  - **Email**: Validates the email format using a regular expression.
- **Appointment Booking**: Allows users to input appointment dates and times, including the extraction of relative date formats like **"Next Monday"** and **"two weeks later"**.
- **Date Extraction**: Uses **dateparser** to extract relative dates like "Next Monday" and converts them into valid **YYYY-MM-DD** format.
- **Input Validation**: Ensures valid phone number and email input, rejecting any invalid formats.

## Requirements

- Python 3.12.6
- Streamlit
- LangChain
- dateparser
- re
- Chroma
- Google Generative AI API (or any LLM for question answering)
- dotenv
- PyPDFLoader (for loading PDF documents)

## Installation

1. Clone this repository:

   ```bash
   https://github.com/HelpRam/Conversational-AI-Assistant-with-Document-QA-and-Booking-Features.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory with your API keys and other necessary configuration.

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. **Upload a PDF**: Once the app is running, upload a PDF document that you want to query.
3. **Ask a Question**: After uploading, enter any query related to the document in the provided text input.
4. **Request a Callback**: Choose the **"Request Callback"** option to fill in your name, phone number, and email for the callback form.
5. **Book an Appointment**: Choose the **"Book Appointment"** option to select a date and time for your appointment.
6. **Handle Errors**: The app will validate the email and phone number inputs and show error messages if the input is invalid.

## Testing

1. **Test File Uploads**: Upload various file formats to ensure only valid PDFs are processed.
2. **Test Invalid Email/Phone Formats**: Ensure the app correctly identifies invalid emails and phone numbers.
3. **Test Date Extraction**: Provide ambiguous date formats like "next Monday" and "in two weeks" to see if the correct date is extracted.


## Code Explanation

- **Document Querying**:
  - The PDF is loaded using `PyPDFLoader`.
  - The document is split into chunks using `RecursiveCharacterTextSplitter`.
  - A vector store (`Chroma`) is used to embed and retrieve documents.
  - **LangChain's** retriever is used to match user queries with document content.

- **Callback Request**:
  - Users are asked to input their name, phone number, and email in sequence.
  - **Regex validation** is used to check the phone number and email format.

- **Appointment Booking**:
  - Dates are extracted using **dateparser**. Relative dates (like "next Monday") are converted into `YYYY-MM-DD` format.
  - Time input is collected after the date is selected.

## Edge Cases Handled

- Invalid file uploads (non-PDF files).
- Invalid email addresses (missing `@`, wrong format).
- Invalid phone numbers (non-10 digit numbers).
- Ambiguous dates ("next Monday", "this Friday", "in two weeks").


## Contact

For any inquiries, feel free to reach out via email at rammey115@gmail.com

