"""
register.py

Student Registration Page
"""

import os
import tempfile

import streamlit as st

from services.api import APIClient


api = APIClient()


def show_register_page():

    st.title("📝 Student Registration")

    st.markdown("---")

    with st.form("register_form"):

        col1, col2 = st.columns(2)

        with col1:

            student_id = st.text_input("Student ID")

            register_number = st.text_input(
                "Register Number"
            )

            name = st.text_input("Student Name")

            department = st.text_input("Department")

        with col2:

            year = st.selectbox(
                "Year",
                [1, 2, 3, 4],
            )

            section = st.text_input("Section")

            phone = st.text_input("Phone Number")

            email = st.text_input("Email")

        st.markdown("### Upload Registration Video")

        video = st.file_uploader(
            "Choose Video",
            type=[
                "mp4",
                "avi",
                "mov",
            ],
        )

        submit = st.form_submit_button(
            "Register Student"
        )

    if submit:

        if video is None:

            st.error(
                "Please upload a registration video."
            )

            return

        student_data = {

            "student_id": student_id,

            "register_number":
            register_number,

            "name":
            name,

            "department":
            department,

            "year":
            year,

            "section":
            section,

            "phone":
            phone,

            "email":
            email,
        }

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        ) as temp_video:

            temp_video.write(
                video.getbuffer()
            )

            temp_path = temp_video.name

        with st.spinner(
            "Registering student..."
        ):

            result = api.register_student(
                student_data,
                temp_path,
            )

        os.remove(temp_path)

        if "message" in result:

            st.success(
                result["message"]
            )

            st.markdown("---")

            st.write(
                f"**Student ID:** {result['student_id']}"
            )

            st.write(
                f"**Frames Extracted:** {result['frames_extracted']}"
            )

            st.write(
                f"**Embeddings Generated:** {result['embeddings_generated']}"
            )

        else:

            st.error(result)