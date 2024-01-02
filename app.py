from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import requests
import os

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#imgtotext
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    print(text)
    return text

#LLM
def story(scenario):
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                            repo_id=model_id, model_kwargs={"temperature": 0.8, "min_length": 50, "max_new_tokens": 100})
    template = """
    you are a story teller;
    you can generate a short story based on simple narrative, the story should be no more than 20 words.
    CONTEXT: {scenario}
    STORY: 
    """
    prompt = PromptTemplate(template=template, input_variables= ['scenario'])
    story_llm = LLMChain(llm= model, prompt=prompt, verbose=True)
    storyline = story_llm.predict(scenario=scenario)
    print(storyline)
    return storyline

#text2speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    payloads = {
         "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open("audio.flac", "wb") as file:
        file.write(response.content)

#Main
def main():
    st.set_page_config(page_title="Image to Audio Story", page_icon="🎙️")
    st.header("Turn an image to an audio story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Your Image', use_column_width=True)
        scenario = img2text(uploaded_file.name)
        storyline = story(scenario)
        text2speech(storyline)

        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(storyline)
        st.audio("audio.flac")

if __name__ == "__main__":
    main()
