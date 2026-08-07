"""
api.py

Handles communication between
Streamlit frontend and FastAPI backend.
"""

import requests


class APIClient:

    def __init__(self):
        self.base_url = "http://127.0.0.1:8000"

    # ----------------------------
    # Register Student
    # ----------------------------

    def register_student(
        self,
        student_data,
        video_path,
    ):

        with open(video_path, "rb") as video:

            files = {
                "video": video
            }

            response = requests.post(
                f"{self.base_url}/register",
                data=student_data,
                files=files,
            )

        return response.json()

    # ----------------------------
    # Identify Student
    # ----------------------------

    def identify_student(
        self,
        image_path,
    ):

        with open(image_path, "rb") as image:

            files = {
                "image": image
            }

            response = requests.post(
                f"{self.base_url}/identify",
                files=files,
            )

        return response.json()

    # ----------------------------
    # Get All Students
    # ----------------------------

    def get_students(self):

        response = requests.get(
            f"{self.base_url}/students"
        )

        return response.json()

    # ----------------------------
    # Get Student
    # ----------------------------

    def get_student(
        self,
        student_id,
    ):

        response = requests.get(
            f"{self.base_url}/students/{student_id}"
        )

        return response.json()

    # ----------------------------
    # Delete Student
    # ----------------------------

    def delete_student(
        self,
        student_id,
    ):

        response = requests.delete(
            f"{self.base_url}/students/{student_id}"
        )

        return response.json()