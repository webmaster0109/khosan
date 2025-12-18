from google import genai
from google.genai import types
import streamlit as st
from utils import get_link_preview

st.set_page_config(page_title="Gemini Thinking Demo", layout="wide")

st.title(":rainbow[Gemini 2.0 Pro w/ Thinking & Search]")

# Sidebar for API Key to avoid hardcoding credentials
with st.sidebar:
    api_key = st.text_input("Google API Key", type="password", placeholder="AIza...")
    st.markdown("[Get an API key](https://aistudio.google.com/app/apikey)")

# Config toggles
col1, col2 = st.columns(2)
with col1:
    thinking_on = st.toggle("Thinking (Reasoning)", value=True, help="Uses gemini-2.0-pro-thinking-exp-01-21")
with col2:
    search_on = st.toggle("Google Search", value=False, help="Enables Grounding with Google Search")

text_input = st.text_input(":rainbow[Enter a prompt]", placeholder="Why is the sky blue?")

if text_input and api_key:
    client = genai.Client(api_key=api_key)
    
    # 1. Select the appropriate model
    # Thinking config is currently only compatible with specific experimental models
    model_id = "gemini-2.5-pro"

    # 2. Build the tools list dynamically
    tools = []
    if search_on:
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    # 3. Construct the config
    config_args = {}
    
    if tools:
        config_args['tools'] = tools
    
    if thinking_on:
        # thinking_config is required to receive thought parts
        config_args['thinking_config'] = types.ThinkingConfig(include_thoughts=True)

    config = types.GenerateContentConfig(**config_args)

    with st.spinner(f"Generating with {model_id}..."):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=text_input,
                config=config
            )

            # 4. Render output
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if not part.text:
                        continue
                    
                    # Check for thought content
                    # Note: SDK behavior for 'thought' attribute may vary slightly by version; 
                    # checking hasattr ensures safety.
                    is_thought = getattr(part, 'thought', False)
                    
                    if is_thought:
                        with st.expander("Thinking Process", expanded=True):
                            st.markdown(part.text)
                    else:
                        st.markdown(part.text)
            
            # Show grounding metadata if available (Search results)
            if search_on and response.candidates[0].grounding_metadata:
                 with st.expander("Search Sources", expanded=False):
                        data = response.candidates[0].grounding_metadata.grounding_chunks
                        url_data = [web.web.uri for web in data]
                        if not url_data:
                          st.info("No search sources available.")
                        else:
                            COLS_PER_ROW = 3  # safe, responsive value
                            for i in range(0, len(url_data), COLS_PER_ROW):
                                cols = st.columns(COLS_PER_ROW)

                                for col, url in zip(cols, url_data[i:i + COLS_PER_ROW]):
                                    with col:
                                        try:
                                            preview = get_link_preview(url)

                                            st.subheader(preview.get("title", "Untitled"))

                                            if preview.get("image"):
                                                st.image(preview["image"], use_container_width=True)

                                            if preview.get("description"):
                                                st.write(preview["description"])
                                            st.markdown(f"[Open link]({preview['url']})")

                                        except Exception:
                                            st.warning("Preview unavailable")
                                            st.markdown(f"[Open link]({url})")
                            

        except Exception as e:
            st.error(f"An error occurred: {e}")

elif text_input and not api_key:
    st.warning("Please enter your API Key in the sidebar to proceed.")
