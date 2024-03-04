import streamlit as st
from streamlit_chat import message as st_message
from audiorecorder import audiorecorder
import time
import json
import requests

import dotenv
from langchain.llms import GooglePalm
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

token_hugging_face = "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"

headers = {"Authorization": f"Bearer {token_hugging_face}"} #TOKEN HUGGING FACE
API_URL_RECOGNITION = "https://api-inference.huggingface.co/models/openai/whisper-tiny.en"

#Voice recognition model
def recognize_speech(audio_file):

    with open(audio_file, "rb") as f:

        data = f.read()

    time.sleep(1)

    while True:
            
        try:

            response = requests.request("POST", API_URL_RECOGNITION, headers=headers, data=data)

            output = json.loads(response.content.decode("utf-8"))

            final_output = output['text']

            break

        except KeyError:

            continue

    return final_output

@st.cache_resource()
def get_model():

    dotenv.load_dotenv()
        
    #Create Google Palm LLM model
    llm = GooglePalm(maxOutputTokens = 200)

    # Initialize instructor embeddings using the Hugging Face model
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large",)

    #Vector DB
    vectordb_file_path = "macchu_picchu_vdb"

    # Load the vector database from the local folder

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


def generate_answer(audio):

    """
    INPUT: QUESTIONS OF TOURIST BY VOICE RECORDING
    
    OUTPUT: INFORMATION OF MACCHU PICCHU
    """

    with st.spinner("Consultation in progress..."):

        
        # To save audio to a file:
        audio.export("./audio.wav", format="wav")
                
        # Voice recognition model
        
        text = recognize_speech("./audio.wav")


        #Response question
        chain = get_model()

        answer_guide = chain.run(text)

        #Save conversation
        st.session_state.history.append({"message": text, "is_user": True})
        st.session_state.history.append({"message": f"{answer_guide}", "is_user": False})


        st.success("Question Answered")            



if __name__ == "__main__":

    # remove the hamburger in the upper right hand corner and the Made with Streamlit footer
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    
    if "history" not in st.session_state:

        st.session_state.history = []


        
  
    st.image("./macchu_picchu.jpg")

 

    st.title("Conversation with a Travel Guide on Machu Picchu")

              
    #Show Input
    audio = audiorecorder("Start recording", "Recording in progress...")

    if len(audio) > 0:

        generate_answer(audio)

        for i, chat in enumerate(st.session_state.history): #Show historical consultation

            st_message(**chat, key =str(i))