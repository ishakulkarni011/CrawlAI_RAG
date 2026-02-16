import os
import torch
from urllib.parse import urlparse
from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

def get_qa_chain(website_url: str, base_dir="vector_db"):
    # 1. Setup paths and vectorstore
    domain = urlparse(website_url).netloc.replace(".", "_")
    persist_dir = os.path.join(base_dir, domain)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"No index found for {website_url}")

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    # 2. Define the Custom Prompt Template
    
    prompt_template = """
            Answer using ONLY the provided website content.
            Rules:
            - Give a brief but complete answer (4–6 sentences).
            - Be clear and informative.
            - Do NOT repeat the question.
            - Do NOT add information not present in the context.
            - If the answer is not found in the context, say so clearly.
            - If the user asks for a link (e.g., GitHub, website), look for it in the "Links Found" section.
            Context:
            {context}
            Question:
            {question}
            Answer:
            """
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    # 3. Setup the Local LLM (FLAN-T5)
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline(
        "text2text-generation", 
        model=model_id,
        tokenizer=tokenizer,
        max_new_tokens=256,
        truncation=True,
        device=0 if torch.cuda.is_available() else -1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    # 4. Initialize RetrievalQA with the Custom Prompt
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}, # This injects your rules
        return_source_documents=True
    )
    return qa