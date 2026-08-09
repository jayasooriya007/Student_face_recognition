"""
frontend.py

Streamlit frontend for Identix.

Behavior:

1. High-confidence match + database record exists
   -> Show register number and student details.

2. High-confidence match + database record does not exist
   -> Show the FAISS register number.
   -> Inform the user that student details are not
      available in the database yet.

3. Similarity below threshold
   -> Do NOT show the candidate register number.
   -> Show a useful low-confidence message.

4. No face / no match
   -> Show an appropriate message.
"""

import requests
import streamlit as st


# ==========================================================
# Configuration
# ==========================================================

API_URL = "http://127.0.0.1:8000"


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Identix",
    page_icon="🎓",
    layout="centered",
)


# ==========================================================
# Header
# ==========================================================

st.title("Identix")

st.subheader(
    "AI Powered Student Identification System"
)

st.write(
    "Upload a student image to identify the student."
)


# ==========================================================
# Image Upload
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload Student Image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


# ==========================================================
# Image Preview
# ==========================================================

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width=350,
    )


# ==========================================================
# Identify Student
# ==========================================================

if st.button(
    "Identify Student",
    type="primary",
    use_container_width=True,
):

    # ======================================================
    # Check Image
    # ======================================================

    if uploaded_file is None:

        st.warning(
            "Please upload a student image first."
        )

    else:

        with st.spinner(
            "Analyzing the image..."
        ):

            try:

                # --------------------------------------------------
                # Reset uploaded file
                # --------------------------------------------------

                uploaded_file.seek(0)

                # --------------------------------------------------
                # Prepare request
                # --------------------------------------------------

                files = {
                    "image": (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type,
                    )
                }

                # --------------------------------------------------
                # Send request to FastAPI
                # --------------------------------------------------

                response = requests.post(
                    f"{API_URL}/identify",
                    files=files,
                    timeout=120,
                )

                # ==================================================
                # HTTP ERROR
                # ==================================================

                if response.status_code != 200:

                    st.error(
                        "Identification request failed."
                    )

                    st.info(
                        "Please try again."
                    )

                else:

                    result = response.json()

                    # ==================================================
                    # Extract Response Values
                    # ==================================================

                    identified = result.get(
                        "identified",
                        False,
                    )

                    message = result.get(
                        "message",
                        "",
                    )

                    register_number = result.get(
                        "register_number"
                    )

                    best_similarity = result.get(
                        "best_similarity"
                    )

                    similarity = result.get(
                        "similarity"
                    )

                    threshold = result.get(
                        "threshold"
                    )

                    student = result.get(
                        "student"
                    )

                    # ==================================================
                    # CASE 1
                    # FULL IDENTIFICATION
                    #
                    # High confidence + MySQL record exists
                    # ==================================================

                    if identified:

                        st.success(
                            "Student identified successfully!"
                        )

                        st.markdown(
                            "### Register Number"
                        )

                        # --------------------------------------------------
                        # Get register number from student object
                        # --------------------------------------------------

                        if student:

                            register_number = (
                                student.get(
                                    "register_number"
                                )
                            )

                        if register_number:

                            st.markdown(
                                f"# {register_number}"
                            )

                        # --------------------------------------------------
                        # Similarity
                        # --------------------------------------------------

                        if best_similarity is not None:

                            st.write(
                                "Face Similarity: "
                                f"{best_similarity:.3f}"
                            )

                        elif similarity is not None:

                            st.write(
                                "Face Similarity: "
                                f"{similarity:.3f}"
                            )

                        # ==================================================
                        # Student Details
                        # ==================================================

                        if student:

                            st.markdown(
                                "### Student Details"
                            )

                            if student.get("name"):

                                st.write(
                                    f"**Name:** "
                                    f"{student.get('name')}"
                                )

                            if student.get("department"):

                                st.write(
                                    f"**Department:** "
                                    f"{student.get('department')}"
                                )

                            if student.get("year"):

                                st.write(
                                    f"**Year:** "
                                    f"{student.get('year')}"
                                )

                            if student.get("section"):

                                st.write(
                                    f"**Section:** "
                                    f"{student.get('section')}"
                                )

                            if student.get("email"):

                                st.write(
                                    f"**Email:** "
                                    f"{student.get('email')}"
                                )

                        else:

                            st.info(
                                "Student details are not "
                                "available in the database yet."
                            )

                    # ==================================================
                    # CASE 2
                    # FAISS FOUND STUDENT
                    # BUT MYSQL RECORD DOES NOT EXIST
                    #
                    # IMPORTANT:
                    # We DO display the register number here.
                    # ==================================================

                    elif (
                        "student information was not found"
                        in message.lower()
                    ):

                        # --------------------------------------------------
                        # FAISS successfully found the student
                        # --------------------------------------------------

                        st.success(
                            "Student identified by "
                            "face recognition!"
                        )

                        st.markdown(
                            "### Register Number"
                        )

                        if register_number:

                            st.markdown(
                                f"# {register_number}"
                            )

                        else:

                            st.markdown(
                                "# Register Number Found"
                            )

                        # --------------------------------------------------
                        # Similarity
                        # --------------------------------------------------

                        if best_similarity is not None:

                            st.write(
                                "Face Similarity: "
                                f"{best_similarity:.3f}"
                            )

                        elif similarity is not None:

                            st.write(
                                "Face Similarity: "
                                f"{similarity:.3f}"
                            )

                        # --------------------------------------------------
                        # Database message
                        # --------------------------------------------------

                        st.info(
                            "Student details are not "
                            "available in the database yet."
                        )

                        st.write(
                            "The face recognition system "
                            "successfully matched the student, "
                            "but the student's detailed record "
                            "has not been added to MySQL yet."
                        )

                    # ==================================================
                    # CASE 3
                    # LOW CONFIDENCE
                    #
                    # DO NOT DISPLAY REGISTER NUMBER
                    # ==================================================

                    elif (
                        "below the current"
                        in message.lower()
                        or
                        "below the"
                        in message.lower()
                        or
                        "sufficiently similar"
                        in message.lower()
                    ):

                        st.warning(
                            "Unable to confidently "
                            "identify the student."
                        )

                        st.markdown(
                            "### Please try another image"
                        )

                        st.info(
                            "The face similarity is below "
                            "the required confidence level."
                        )

                        # --------------------------------------------------
                        # Useful instructions
                        # --------------------------------------------------

                        st.markdown(
                            "### 💡 For better recognition"
                        )

                        st.markdown(
                            """
                            - Upload a clearer image
                            - Make sure the face is clearly visible
                            - Use good lighting
                            - Avoid heavily blurred images
                            - Avoid very dark images
                            - Keep the face reasonably close to the camera
                            - Try to keep the face looking toward the camera
                            """
                        )

                        # --------------------------------------------------
                        # Show similarity only
                        #
                        # DO NOT show register number
                        # --------------------------------------------------

                        if best_similarity is not None:

                            st.write(
                                "Similarity: "
                                f"{best_similarity:.3f}"
                            )

                        elif similarity is not None:

                            st.write(
                                "Similarity: "
                                f"{similarity:.3f}"
                            )

                        if threshold is not None:

                            st.write(
                                "Required confidence: "
                                f"{threshold:.3f}"
                            )

                    # ==================================================
                    # CASE 4
                    # NO FACE / NO MATCH / OTHER FAILURE
                    # ==================================================

                    else:

                        st.error(
                            "Student could not be identified."
                        )

                        if message:

                            st.write(
                                message
                            )

                        st.markdown(
                            "### 💡 Please try again"
                        )

                        st.markdown(
                            """
                            - Upload an image containing a clear face
                            - Make sure the face is visible
                            - Use sufficient lighting
                            - Avoid heavy blur
                            - Avoid extreme face angles
                            """
                        )

                        if best_similarity is not None:

                            st.write(
                                "Best Similarity: "
                                f"{best_similarity:.3f}"
                            )

                        elif similarity is not None:

                            st.write(
                                "Similarity: "
                                f"{similarity:.3f}"
                            )


            # ======================================================
            # CONNECTION ERROR
            # ======================================================

            except requests.exceptions.ConnectionError:

                st.error(
                    "Unable to connect to the "
                    "Identix backend."
                )

                st.info(
                    "Please make sure FastAPI is running."
                )

                st.code(
                    "uvicorn app.api.main:app --reload"
                )


            # ======================================================
            # TIMEOUT
            # ======================================================

            except requests.exceptions.Timeout:

                st.error(
                    "The identification request "
                    "took too long."
                )

                st.info(
                    "Please try again."
                )


            # ======================================================
            # OTHER ERROR
            # ======================================================

            except Exception as e:

                st.error(
                    "An unexpected error occurred."
                )

                st.exception(e)


# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "Identix • AI Powered Student Identification System"
)