"""
frontend.py

Streamlit frontend for Identix.
AI Powered Student Identification System.
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

                    try:

                        error_data = response.json()

                        if "detail" in error_data:

                            st.write(
                                error_data["detail"]
                            )

                    except Exception:

                        pass

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
                    # GROUP PHOTO IDENTIFICATION
                    #
                    # Display information for every identified face.
                    # ==================================================

                    students = result.get("results")

                    total_faces = result.get("total_faces")
                    identified_students = result.get(
                        "identified_faces"
                    )

                    if isinstance(students, list):

                        if total_faces is not None:
                            st.success(
                                f"Processed {total_faces} face(s). "
                                f"Identified "
                                f"{identified_students if identified_students is not None else len(students)} "
                                f"student(s)."
                            )
                        else:
                            st.success(
                                f"Identified {len(students)} student(s)."
                            )

                        # --------------------------------------------------
                        # Display every identified student
                        # --------------------------------------------------

                        for index, face_result in enumerate(
                            students,
                            start=1
                        ):

                            # --------------------------------------------------
                            # Some backend responses may contain the
                            # student data directly, while others may
                            # contain it inside a "student" object.
                            # --------------------------------------------------

                            if isinstance(face_result, dict):

                                student_data = face_result.get(
                                    "student"
                                )

                                if not isinstance(
                                    student_data,
                                    dict
                                ):
                                    student_data = face_result

                                face_register_number = (
                                    face_result.get(
                                        "register_number"
                                    )
                                    or student_data.get(
                                        "register_number"
                                    )
                                )

                                face_best_similarity = (
                                    face_result.get(
                                        "best_similarity"
                                    )
                                )

                                face_similarity = (
                                    face_result.get(
                                        "similarity"
                                    )
                                )

                                face_message = face_result.get(
                                    "message",
                                    ""
                                )

                                face_identified = face_result.get(
                                    "identified",
                                    True
                                )

                            else:

                                student_data = {}

                                face_register_number = None
                                face_best_similarity = None
                                face_similarity = None
                                face_message = ""
                                face_identified = False

                            st.markdown(
                                "---"
                            )

                            st.markdown(
                                f"### Face {index}"
                            )

                            # --------------------------------------------------
                            # Unknown / unsuccessful face
                            # --------------------------------------------------

                            if (
                                not face_identified
                                or not student_data
                            ):

                                st.warning(
                                    "Student could not be "
                                    "identified for this face."
                                )

                                if face_message:
                                    st.write(
                                        face_message
                                    )

                                if (
                                    face_best_similarity
                                    is not None
                                ):
                                    st.write(
                                        "Similarity: "
                                        f"{face_best_similarity:.3f}"
                                    )

                                elif (
                                    face_similarity
                                    is not None
                                ):
                                    st.write(
                                        "Similarity: "
                                        f"{face_similarity:.3f}"
                                    )

                                continue

                            # --------------------------------------------------
                            # Register Number
                            # --------------------------------------------------

                            student_register_number = (
                                face_register_number
                                or student_data.get(
                                    "register_number"
                                )
                            )

                            if student_register_number:

                                st.markdown(
                                    "### Register Number"
                                )

                                st.markdown(
                                    f"# "
                                    f"{student_register_number.upper()}"
                                )

                            # --------------------------------------------------
                            # Similarity
                            # --------------------------------------------------

                            if (
                                face_best_similarity
                                is not None
                            ):

                                st.write(
                                    "Face Similarity: "
                                    f"{face_best_similarity:.3f}"
                                )

                            elif (
                                face_similarity
                                is not None
                            ):

                                st.write(
                                    "Face Similarity: "
                                    f"{face_similarity:.3f}"
                                )

                            # ==================================================
                            # Student Details
                            # ==================================================

                            st.markdown(
                                "### Student Details"
                            )

                            # --------------------------------------------------
                            # Name
                            # --------------------------------------------------

                            if student_data.get(
                                "student_name"
                            ):

                                st.write(
                                    f"**Name:** "
                                    f"{student_data.get('student_name')}"
                                )

                            # --------------------------------------------------
                            # Department
                            # --------------------------------------------------

                            if student_data.get(
                                "department"
                            ):

                                st.write(
                                    f"**Department:** "
                                    f"{student_data.get('department')}"
                                )

                            # --------------------------------------------------
                            # Academic Batch
                            # --------------------------------------------------

                            if student_data.get(
                                "academic_batch"
                            ):

                                st.write(
                                    f"**Academic Batch:** "
                                    f"{student_data.get('academic_batch')}"
                                )

                            # --------------------------------------------------
                            # Year
                            # --------------------------------------------------

                            if student_data.get(
                                "year"
                            ) is not None:

                                st.write(
                                    f"**Year:** "
                                    f"{student_data.get('year')}"
                                )

                            # --------------------------------------------------
                            # Section
                            # --------------------------------------------------

                            if student_data.get(
                                "section"
                            ):

                                st.write(
                                    f"**Section:** "
                                    f"{student_data.get('section')}"
                                )

                            # --------------------------------------------------
                            # College Email
                            # --------------------------------------------------

                            if student_data.get(
                                "college_email"
                            ):

                                st.write(
                                    f"**College Email:** "
                                    f"{student_data.get('college_email')}"
                                )

                            # --------------------------------------------------
                            # Personal Email
                            # --------------------------------------------------

                            if student_data.get(
                                "personal_email"
                            ):

                                st.write(
                                    f"**Personal Email:** "
                                    f"{student_data.get('personal_email')}"
                                )

                            # --------------------------------------------------
                            # Student Phone
                            # --------------------------------------------------

                            if student_data.get(
                                "phone_number"
                            ):

                                st.write(
                                    f"**Student Mobile:** "
                                    f"{student_data.get('phone_number')}"
                                )

                            # ==================================================
                            # Parent / Guardian Details
                            # ==================================================

                            st.markdown(
                                "### Parent / Guardian Details"
                            )

                            if student_data.get(
                                "parent_guardian_name"
                            ):

                                st.write(
                                    f"**Parent / Guardian Name:** "
                                    f"{student_data.get('parent_guardian_name')}"
                                )

                            if student_data.get(
                                "parent_guardian_mobile"
                            ):

                                st.write(
                                    f"**Parent / Guardian Mobile:** "
                                    f"{student_data.get('parent_guardian_mobile')}"
                                )

                            # ==================================================
                            # Mentor Details
                            # ==================================================

                            st.markdown(
                                "### Mentor Details"
                            )

                            if student_data.get(
                                "mentor_name"
                            ):

                                st.write(
                                    f"**Mentor Name:** "
                                    f"{student_data.get('mentor_name')}"
                                )

                            if student_data.get(
                                "mentor_phone"
                            ):

                                st.write(
                                    f"**Mentor Mobile:** "
                                    f"{student_data.get('mentor_phone')}"
                                )

                            # ==================================================
                            # Status
                            # ==================================================

                            if student_data.get(
                                "status"
                            ):

                                st.write(
                                    f"**Student Status:** "
                                    f"{student_data.get('status')}"
                                )

                            # ==================================================
                            # Recognition Information
                            # ==================================================

                            if threshold is not None:

                                st.caption(
                                    f"Required confidence: "
                                    f"{threshold:.3f}"
                                )

                    # ==================================================
                    # CASE 1
                    # FULL IDENTIFICATION
                    #
                    # FAISS matched + MySQL record exists
                    # ==================================================

                    elif identified and student:

                        st.success(
                            "Student identified successfully!"
                        )

                        # --------------------------------------------------
                        # Register Number
                        # --------------------------------------------------

                        st.markdown(
                            "### Register Number"
                        )

                        student_register_number = (
                            student.get(
                                "register_number"
                            )
                            or register_number
                        )

                        if student_register_number:

                            st.markdown(
                                f"# {student_register_number.upper()}"
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

                        st.markdown(
                            "### Student Details"
                        )

                        # --------------------------------------------------
                        # Name
                        # --------------------------------------------------

                        if student.get("student_name"):

                            st.write(
                                f"**Name:** "
                                f"{student.get('student_name')}"
                            )

                        # --------------------------------------------------
                        # Department
                        # --------------------------------------------------

                        if student.get("department"):

                            st.write(
                                f"**Department:** "
                                f"{student.get('department')}"
                            )

                        # --------------------------------------------------
                        # Academic Batch
                        # --------------------------------------------------

                        if student.get("academic_batch"):

                            st.write(
                                f"**Academic Batch:** "
                                f"{student.get('academic_batch')}"
                            )

                        # --------------------------------------------------
                        # Year
                        # --------------------------------------------------

                        if student.get("year") is not None:

                            st.write(
                                f"**Year:** "
                                f"{student.get('year')}"
                            )

                        # --------------------------------------------------
                        # Section
                        # --------------------------------------------------

                        if student.get("section"):

                            st.write(
                                f"**Section:** "
                                f"{student.get('section')}"
                            )

                        # --------------------------------------------------
                        # College Email
                        # --------------------------------------------------

                        if student.get("college_email"):

                            st.write(
                                f"**College Email:** "
                                f"{student.get('college_email')}"
                            )

                        # --------------------------------------------------
                        # Personal Email
                        # --------------------------------------------------

                        if student.get("personal_email"):

                            st.write(
                                f"**Personal Email:** "
                                f"{student.get('personal_email')}"
                            )

                        # --------------------------------------------------
                        # Student Phone
                        # --------------------------------------------------

                        if student.get("phone_number"):

                            st.write(
                                f"**Student Mobile:** "
                                f"{student.get('phone_number')}"
                            )

                        # ==================================================
                        # Parent / Guardian Details
                        # ==================================================

                        st.markdown(
                            "### Parent / Guardian Details"
                        )

                        if student.get(
                            "parent_guardian_name"
                        ):

                            st.write(
                                f"**Parent / Guardian Name:** "
                                f"{student.get('parent_guardian_name')}"
                            )

                        if student.get(
                            "parent_guardian_mobile"
                        ):

                            st.write(
                                f"**Parent / Guardian Mobile:** "
                                f"{student.get('parent_guardian_mobile')}"
                            )

                        # ==================================================
                        # Mentor Details
                        # ==================================================

                        st.markdown(
                            "### Mentor Details"
                        )

                        if student.get("mentor_name"):

                            st.write(
                                f"**Mentor Name:** "
                                f"{student.get('mentor_name')}"
                            )

                        if student.get("mentor_phone"):

                            st.write(
                                f"**Mentor Mobile:** "
                                f"{student.get('mentor_phone')}"
                            )

                        # ==================================================
                        # Status
                        # ==================================================

                        if student.get("status"):

                            st.write(
                                f"**Student Status:** "
                                f"{student.get('status')}"
                            )

                        # ==================================================
                        # Recognition Information
                        # ==================================================

                        if threshold is not None:

                            st.caption(
                                f"Required confidence: "
                                f"{threshold:.3f}"
                            )


                    # ==================================================
                    # CASE 2
                    # FAISS FOUND STUDENT
                    # BUT MYSQL RECORD DOES NOT EXIST
                    #
                    # IMPORTANT:
                    # Display the register number.
                    # ==================================================

                    elif (
                        "student information was not found"
                        in message.lower()
                        or
                        "student information"
                        in message.lower()
                    ):

                        st.success(
                            "Student identified by "
                            "face recognition!"
                        )

                        # --------------------------------------------------
                        # Register Number
                        # --------------------------------------------------

                        st.markdown(
                            "### Register Number"
                        )

                        if register_number:

                            st.markdown(
                                f"# {register_number.upper()}"
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
                        # Database Information
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
                        or
                        "confidence"
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
                        # Useful Instructions
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