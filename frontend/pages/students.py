"""
students.py

Student Management Page
"""

import streamlit as st
import pandas as pd

from services.api import APIClient


api = APIClient()


def show_students_page():

    st.title("👨‍🎓 Student Management")

    st.markdown("---")

    students = api.get_students()

    if not students:

        st.info("No students registered.")

        return

    df = pd.DataFrame(students)

    search = st.text_input(
        "Search Student",
        placeholder="Enter Student ID, Name, Register Number..."
    )

    if search:

        search = search.lower()

        df = df[
            df.astype(str)
              .apply(
                  lambda x: x.str.lower()
              )
              .apply(
                  lambda x: x.str.contains(search)
              )
              .any(axis=1)
        ]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    student_ids = df["student_id"].tolist()

    selected_student = st.selectbox(
        "Select Student",
        student_ids
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("View Details"):

            student = api.get_student(
                selected_student
            )

            st.markdown("---")

            st.subheader("Student Details")

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

    with col2:

        if st.button("Delete Student"):

            result = api.delete_student(
                selected_student
            )

            st.success(
                result["message"]
            )

            st.rerun()