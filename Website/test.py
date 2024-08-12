import streamlit as st
from pathlib import Path
import os

os.getcwd()

st.write(os.getcwd())

photo = Path(__file__).parents[1] / "IMG_0701.JPG"

st.write(photo)
st.image("/mount/src/phd/IMG_0701.JPG", use_column_width=True,
         caption="This is the second image caption, providing details about the photo.")
