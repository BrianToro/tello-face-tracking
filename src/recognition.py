import os
import face_recognition
import numpy as np
from numpy.typing import NDArray

TARGET_IMAGES_DIR = "./assets/target_images"
FaceEmbedding = NDArray[np.float64]


def load_image() -> FaceEmbedding:
    # load images from dir
    temp_embeddings = []
    for image_path in os.listdir(TARGET_IMAGES_DIR):
        if ".gitkeep" in image_path:
            continue

        image = face_recognition.load_image_file(
            os.path.join(TARGET_IMAGES_DIR, image_path)
        )
        try:
            embedding = face_recognition.face_encodings(image)[0]
            temp_embeddings.append(embedding)
        except IndexError:
            print(f"{image_path} is not a valid image")

    if not temp_embeddings:
        raise Exception("No faces found")

    # calc the average
    embedding_average = np.mean(temp_embeddings, axis=0)
    return np.asarray(embedding_average, dtype=np.float64)


def locate_faces(rgb_frame, known_face_encodings, known_face_names):
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names: list[str] = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=0.5
        )
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    return face_names, face_locations
