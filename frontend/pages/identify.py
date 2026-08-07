"""
identify.py

Student Identification Page
"""

import os
import tempfile

import streamlit as st

from services.api import APIClient


api = APIClient()


def show_identify_page():

    st.title("🔍 Student Identification")

    st.markdown("---")

    uploaded_image = st.file_uploader(
        "Upload Student Image",
        type=["jpg", "jpeg", "png"],
    )

    if st.button("Identify Student"):

        if uploaded_image is None:
            st.error("Please upload an image.")
            return

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
        ) as temp_image:

            temp_image.write(
                uploaded_image.getbuffer()
            )

            temp_path = temp_image.name

        with st.spinner(
            "Identifying student..."
        ):

            result = api.identify_student(
                temp_path
            )

        os.remove(temp_path)

        st.markdown("---")

        if result.get("success"):

            st.success(result["message"])

            similarity = result["similarity"]

            st.subheader("Recognition Confidence")

            st.progress(
                min(similarity, 1.0)
            )

            st.write(
                f"Similarity Score: **{similarity:.4f}**"
            )

            st.markdown("---")

            student = result["student"]

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**Student ID:** {student['student_id']}"
                )

                st.write(
                    f"**Register Number:** {student['register_number']}"
                )

                st.write(
                    f"**Name:** {student['name']}"
                )

                st.write(
                    f"**Department:** {student['department']}"
                )

            with col2:

                st.write(
                    f"**Year:** {student['year']}"
                )

                st.write(
                    f"**Section:** {student['section']}"
                )

                st.write(
                    f"**Phone:** {student['phone']}"
                )

                st.write(
                    f"**Email:** {student['email']}"
                )

        else:

            st.error(result["message"])

            if "similarity" in result:

                st.write(
                    f"Similarity Score: {result['similarity']:.4f}"
                )