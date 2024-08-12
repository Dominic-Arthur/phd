import streamlit as st
import json


st.image("./Website/Photos/IMG_0701.JPG", use_column_width=True,
         caption="This is the second image caption, providing details about the photo.")

with open("./Website/images.json", "r") as f:
    data = json.load(f)
    st.image(data["images"]["path"], use_column_width=True, caption=data["images"]["caption"])
