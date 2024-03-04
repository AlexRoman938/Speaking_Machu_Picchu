import dotenv
from langchain.llms import GooglePalm
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

#Create Google Palm LLM model
llm = GooglePalm(maxOutputTokens = 200)

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",)

#Vector DB
vectordb_file_path = "macchu_picchu_vdb"

def get_qa_chain():
    # Load the vector database from the local folder

    dotenv.load_dotenv()

    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold = 0.7) #you can change score_threshold

    prompt_template = """
    As a travel guide expert on Machu Picchu, given the following context and a question, generate a conversational response following a conversation with an answer based solely on this context.
    In the answer try to provide as much text as possible from "response" section in the source document context.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    {chat_history}
    Human: {question}
    Assistant:"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "chat_history", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    memory = ConversationBufferMemory(memory_key = f"chat_history", return_messages = True)


    chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                chain_type="stuff",
                                retriever=retriever,
                                memory=memory, combine_docs_chain_kwargs=chain_type_kwargs)
    
    return chain

if __name__ == "__main__":

    chain = get_qa_chain()
