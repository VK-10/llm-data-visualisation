import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import LangchainLLM
from langchain_ollama import ChatOllama

# Create LangChain LLM wrapper
chat_llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.2,
)
pandasai_llm = LangchainLLM(langchain_llm=chat_llm)

st.title("Data Analysis with PandasAI")

uploader_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploader_file is not None:
    data = pd.read_csv(uploader_file)
    st.write("### Uploaded Data")
    st.write(data.head(6))
    
    # Create SmartDataframe with proper LLM configuration
    df = SmartDataframe(data, config={"llm": pandasai_llm})
    
    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                try:
                    response = df.chat(prompt)
                    st.markdown("### LLM Response")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a prompt!")
else:
    st.info("Please upload a CSV file to get started.")
