"""
app.py

Main entry point for the Identix Streamlit application.
"""

import streamlit as st

st.set_page_config(
    page_title="Identix",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.title("🎓 Identix")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Register Student",
        "Identify Student",
        "Student Management",
    ],
)

# -----------------------------------
# Dashboard
# -----------------------------------

if page == "Dashboard":

    st.title("🎓 Identix")

    st.subheader(
        "AI Powered Student Identification System"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            ### Features

            - Student Registration
            - Face Recognition
            - FAISS Search
            - PostgreSQL Database
            - FastAPI Backend
            """
        )

    with col2:

        st.success(
            """
            ### Technologies

            - InsightFace
            - ArcFace
            - SCRFD
            - FAISS
            - FastAPI
            - Streamlit
            """
        )

    st.markdown("---")

    st.write(
        """
        Welcome to **Identix**.

        Use the navigation menu on the left to:

        - Register new students
        - Identify students
        - View registered students
        """
    )

# -----------------------------------
# Register
# -----------------------------------

elif page == "Register Student":

    from pages.register import show_register_page

    show_register_page()

# -----------------------------------
# Identify
# -----------------------------------

elif page == "Identify Student":

    from pages.identify import show_identify_page

    show_identify_page()

# -----------------------------------
# Student Management
# -----------------------------------

elif page == "Student Management":

    from pages.students import show_students_page

    show_students_page()