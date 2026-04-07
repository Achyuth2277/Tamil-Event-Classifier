"""Simple Streamlit UI for Tamil event classification."""

from __future__ import annotations

import streamlit as st

from predict import predict_event


st.set_page_config(page_title="Tamil Event Classifier", page_icon=":newspaper:", layout="centered")

st.title("Tamil Event Classifier")
st.write("Enter a Tamil sentence to get the event category and a short explanation in English.")

sample_text = "தமிழ்நாட்டில் கடும் மழை பெய்தது"
tamil_text = st.text_area(
    "Tamil Sentence",
    value=sample_text,
    height=120,
    placeholder="Type a Tamil sentence here...",
)

predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    if not tamil_text.strip():
        st.warning("Please enter a Tamil sentence.")
    else:
        result = predict_event(tamil_text)

        st.subheader("Prediction Result")
        st.write(f'**Input:** {result["tamil_text"]}')
        st.write(f'**English Translation:** {result["english_translation"] or "Unavailable"}')
        st.write(f'**Predicted Category:** {result["predicted_category"]}')
        st.write(f'**Event Type:** {result["event_subtype"]}')
        st.write(f'**Info:** {result["event_information"]}')

        if "confidence_score" in result:
            st.write(f'**Confidence:** {result["confidence_score"]}')

        if result["translation_note"] != "Translation generated successfully.":
            st.info(result["translation_note"])

        if result["confidence_note"] != "Prediction generated successfully.":
            st.info(result["confidence_note"])

st.caption("Classification runs on Tamil text directly. Translation is optional and may be unavailable in this environment.")
