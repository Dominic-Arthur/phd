import streamlit as st
from pathlib import Path

photo = Path(__file__).parents[1] / "IMG_0701.JPG"

st.image(photo, use_column_width=True,
         caption="This is the second image caption, providing details about the photo.")
