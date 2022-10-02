import streamlit as st
import time
import numpy as np
import streamlit.components.v1 as components

st.set_page_config(page_title="Plotting Demo", page_icon="📈")


p = open("ai_papers_plot3.html")
components.html(p.read(),height = 500,width=1000)