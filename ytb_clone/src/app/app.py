import streamlit as st
from openai import OpenAI
import json
from services import import_video_stream, get_stream_response
import time

st.session_state['urls'] = ["https://www.youtube.com/watch?v=EDj-Xo8AlSU&t=120s", "https://www.youtube.com/watch?v=k5mJgmtRXZA"]

def main():
    st.title("Youtube Video RAG")
    with st.expander("Import Youtube's Video"):
        url = st.text_input("Video URL")
        import_btn = st.button("Import")
        
    if import_btn:
        if url in st.session_state['urls']:
            st.warning("URL has been imported")
        else:
            progress_text = "Processing"
            
            events = import_video_stream(url)
            
            process_bar = st.progress(0, text=progress_text)
            
            percent_complete = 0
            for event in events:
                message = json.loads(event)["message"]
                percent_complete += 17
                
                percent_complete = min(100, percent_complete)
                
                process_bar.progress(percent_complete, text=message)
                time.sleep(2)
                
            # Initialization
            if 'urls' not in st.session_state:
                st.session_state['urls'] = [url]
            else:
                st.session_state['urls'].append(url)
                
            process_bar.empty()
            
    url_select = st.selectbox("Youtube url", options= st.session_state.get('urls', ""))
                    
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "max_messages" not in st.session_state:
        # Counting both user and assistant messages, so 10 rounds of conversation
        st.session_state.max_messages = 1000

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if len(st.session_state.messages) >= st.session_state.max_messages:
        st.session_state.max_messages = 1000
        st.session_state.messages = []
        st.info(
            """Notice: The maximum message limit for this demo version has been reached. We value your interest!
            We encourage you to experience further interactions by building your own application with instructions
            from Streamlit's [Build a basic LLM chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)
            tutorial. Thank you for your understanding."""
        )

    else:
        if prompt := st.chat_input("What is up?"):
            print(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant") as c:
                try:
                    video_id = url_select.split("v=")[1]
                    video_id = video_id.split("&")[0]                    
                    stream = get_stream_response(prompt, video_id)
                    response = st.write_stream(stream)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    print(e)
                    st.session_state.max_messages = len(st.session_state.messages)
                    rate_limit_message = """
                        Oops! Sorry, I can't talk now. Too many people have used
                        this service recently.
                    """
                    st.session_state.messages.append(
                        {"role": "assistant", "content": rate_limit_message}
                    )
                    st.rerun()

             
if __name__ == "__main__":
    main()