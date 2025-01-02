import streamlit as st
import dateparser
import re
from datetime import timedelta
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit title
st.title("Conversational AI Chatbot with Document QA and Booking Features")

# Initialize session state for callback process and appointment booking
if "callback_state" not in st.session_state:
    st.session_state.callback_state = None
    st.session_state.user_details = {}

if "appointment_state" not in st.session_state:
    st.session_state.appointment_state = None
    st.session_state.appointment_details = {}

# File Upload
uploaded_file = st.file_uploader("Upload your document (PDF)", type=["pdf"])
if uploaded_file is not None:
    try:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and process document
        loader = PyPDFLoader("uploaded_file.pdf")
        data = loader.load()
        st.success("PDF loaded successfully!")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        st.stop()
else:
    st.info("Please upload a PDF file to proceed.")
    st.stop()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Initialize Vector Store
try:
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory=".chroma_db"
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
except Exception as e:
    st.error(f"Error initializing vector store: {e}")
    st.stop()

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)
except Exception as e:
    st.error(f"Error initializing LLM: {e}")
    st.stop()

# Define system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer: "
    "If you don't know the answer, say that you don't know. Be concise."
    "\n\n"
    "{context}"
)

# Date parsing function to convert relative date (e.g., "next Monday") to a complete date format (YYYY-MM-DD)
def extract_date_from_input(user_input):
    parsed_date = dateparser.parse(user_input, settings={'PREFER_DATES_FROM': 'future'})
    
    if parsed_date:
        return parsed_date.date()
    
    if "next" in user_input.lower():
        try:
            day_of_week = user_input.lower().split("next")[-1].strip()
            today = dateparser.parse("today")
            weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            day_index = weekday_names.index(day_of_week)
            days_ahead = (day_index - today.weekday()) % 7
            next_day = today + timedelta(days=days_ahead)
            return next_day.date()
        except Exception as e:
            return None
    return None

# Validate email using regular expression
def validate_email(email):
    email_regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    return re.match(email_regex, email)

# Validate phone number (simple validation for 10 digits)
def validate_phone(phone):
    phone_regex = r"^\d{10}$"
    return re.match(phone_regex, phone)

# Tool functions for LangChain Agents
def validate_email_tool(email):
    if validate_email(email):
        return "Valid email"
    return "Invalid email"

def validate_phone_tool(phone):
    if validate_phone(phone):
        return "Valid phone"
    return "Invalid phone"

def parse_date_tool(input_date):
    parsed_date = extract_date_from_input(input_date)
    if parsed_date:
        return str(parsed_date)
    return "Invalid date"

# Define Tools
tools = [
    Tool(name="Validate Email", func=validate_email_tool, description="Validates an email address."),
    Tool(name="Validate Phone", func=validate_phone_tool, description="Validates a phone number."),
    Tool(name="Parse Date", func=parse_date_tool, description="Parses natural language dates into YYYY-MM-DD."),
]

# Initialize Agent
agent = initialize_agent(tools=tools, llm=llm, agent_type="zero-shot-react-description")

# Logic for Callback Request with Tool-Agent Integration
def handle_callback():
    if st.session_state.callback_state is None:
        st.session_state.callback_state = "name"
        st.session_state.user_details = {}

    if st.session_state.callback_state == "name":
        query = st.text_input("Please enter your name:", key="name_input")
        if query:
            st.session_state.user_details["name"] = query
            st.session_state.callback_state = "phone"

    if st.session_state.callback_state == "phone":
        query = st.text_input("Please enter your phone number:", key="phone_input")
        if query:
            result = agent.run({"tool": "Validate Phone", "input": query})
            if "Valid phone" in result:
                st.session_state.user_details["phone"] = query
                st.session_state.callback_state = "email"
            else:
                st.error("Invalid phone number. Please enter a valid 10-digit phone number.")

    if st.session_state.callback_state == "email":
        query = st.text_input("Please enter your email address:", key="email_input")
        if query:
            result = agent.run({"tool": "Validate Email", "input": query})
            if "Valid email" in result:
                st.session_state.user_details["email"] = query
                st.success(
                    f"Thank you, {st.session_state.user_details['name']}. "
                    f"We will contact you at {st.session_state.user_details['phone']} "
                    f"or via {st.session_state.user_details['email']}."
                )
                # Reset state
                st.session_state.callback_state = None
                st.session_state.user_details = {}
            else:
                st.error("Invalid email address. Please enter a valid email address.")

# Logic for Booking Appointment with Tool-Agent Integration
def handle_appointment():
    if st.session_state.appointment_state is None:
        st.session_state.appointment_state = "date"
        st.session_state.appointment_details = {}

    if st.session_state.appointment_state == "date":
        query = st.text_input("Please enter the date for your appointment (e.g., 'next Monday'):", key="date_input")
        if query:
            result = agent.run({"tool": "Parse Date", "input": query})
            if "Invalid date" not in result:
                st.session_state.appointment_details["date"] = result
                st.session_state.appointment_state = "time"
            else:
                st.error("Invalid date. Please provide a valid date or relative date (e.g., 'next Monday').")

    if st.session_state.appointment_state == "time":
        query = st.text_input(f"Please enter the time for your appointment on {st.session_state.appointment_details['date']}:", key="time_input")
        if query:
            st.session_state.appointment_details["time"] = query
            st.success(
                f"Your appointment has been scheduled for {st.session_state.appointment_details['date']} "
                f"at {st.session_state.appointment_details['time']}."
            )
            # Reset state
            st.session_state.appointment_state = None
            st.session_state.appointment_details = {}

# Select action type
action = st.selectbox(
    "Choose an action", 
    options=["Answer a Question", "Request Callback", "Book Appointment"]
)

# Handle callback and appointment flows based on selected action
if action == "Request Callback":
    handle_callback()

elif action == "Book Appointment":
    handle_appointment()

elif action == "Answer a Question":
    # QA Mode (Handling queries from the uploaded document)
    query = st.text_input("Ask a question:")
    if query:
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}")
                ]
            )
            question_answer_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=prompt,
                document_variable_name="context",  # Ensure the variable for documents matches
            )
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            response = rag_chain.invoke({"input": query})
            st.write(response["answer"])
        except Exception as e:
            st.error(f"Error while processing the query: {e}")
