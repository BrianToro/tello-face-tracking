import cv2
import argparse
import recognition
from djitellopy import Tello

def load_args():
    parser = argparse.ArgumentParser(description="Make the drone follow your face")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="The name of the target that the drone will follow"
    )
    parser.add_argument(
        "--video-source",
        type=str,
        required=True,
        help="Choose a video source between drone and webcam"
    )
    return parser.parse_args()

def main():
    # load data to capture the target
    args = load_args()
    target_name = args.target
    video_source = args.video_source
    face_embedding = recognition.load_image()
    known_face_encodings = [face_embedding]
    known_face_names = [target_name]

    if video_source == "webcam":
        process_with_internal_video(known_face_encodings, known_face_names, target_name)
    elif video_source == "drone":
        process_with_drone_video(known_face_encodings, known_face_names, target_name)
    else:
        print("Unknown video source")

def process_with_internal_video(known_face_encodings, known_face_names, target_name):
    video_capture = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_names, face_locations = recognition.locate_faces(
            rgb_small_frame,
            known_face_encodings,
            known_face_names,
        )

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (36, 125, 60) if name == target_name else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
            )

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("Successfully finished program.")

def process_with_drone_video(known_face_encodings, known_face_names, target_name):
    stride = 2
    tdrone = Tello()
    tdrone.connect()
    tdrone.streamon()
    frame_read = tdrone.get_frame_read()

    frame_idx = 0
    while True:
        frame = frame_read.frame  # BGR (numpy array)
        if frame is None or frame.size == 0:
            continue

        frame_idx += 1


        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_names, face_locations = recognition.locate_faces(
            rgb_small, known_face_encodings, known_face_names
        )

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            color = (36, 125, 60) if name == target_name else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow("Tello Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    tdrone.streamoff()
    tdrone.end()


if __name__ == "__main__":
    main()
