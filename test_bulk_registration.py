from app.services.bulk_registration_service import (
    BulkRegistrationService,
)


service = BulkRegistrationService()

folders = service.get_student_folders()

print("=" * 60)

print(
    "Total Student Folders:",
    len(folders)
)

print("=" * 60)


for folder in folders:

    print()
    print("Student:", folder.name)

    result = service.process_student(
        folder
    )

    # ------------------------------------------
    # No Video
    # ------------------------------------------

    if result["status"] == "NO_VIDEO":

        print("  ❌ No registration video")

        continue

    # ------------------------------------------
    # No Extracted Frames
    # ------------------------------------------

    if (
        result["status"]
        == "NO_PROCESSED_FRAMES"
    ):

        print(
            "  ❌ No previously extracted frames"
        )

        print(
            "  Run frame extraction for this student."
        )

        continue

    # ------------------------------------------
    # Successful Processing
    # ------------------------------------------

    print(
        "  Video:",
        result["video"]
    )

    print(
        "  Existing frames checked:",
        result["total_frames_checked"]
    )

    print(
        "  Good frames:",
        result["good_frames"]
    )

    print(
        "  Rejected frames:",
        result["rejected_frames"]
    )

    print(
        "  Saved to:",
        result["quality_filtered_folder"]
    )


print()
print("=" * 60)

print(
    "Quality filtering completed."
)

print("=" * 60)