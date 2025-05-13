import streamlit as st
import pickle
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="Fake News Classifier", page_icon="📰")
