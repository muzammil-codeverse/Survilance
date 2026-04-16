import streamlit as st

st.set_page_config(page_title="Violence Detection", layout="centered")

st.title("Violence Detection System")
st.write("Upload a video to analyze.")

video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if video_file:
    st.video(video_file)
    st.info("Processing will be added in Phase 8")
