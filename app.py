import streamlit as st
from streamlit_chat import message as st_message
from audiorecorder import audiorecorder
import time
import json
import requests

from langchain_helper import get_qa_chain


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
        chain = get_qa_chain()

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