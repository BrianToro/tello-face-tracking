import cv2
import argparse
from typing import Optional
import recognition
from djitellopy import Tello

FACE_WIDTH_METERS = 0.16  # Average human face width used for distance estimation
WEBCAM_FOCAL_LENGTH_PX = 600.0  # Approximate focal length in pixels for the default webcam


def estimate_depth_m(bbox_width: int, focal_px: float = WEBCAM_FOCAL_LENGTH_PX) -> Optional[float]:
    if bbox_width <= 0:
        return None
    return (FACE_WIDTH_METERS * focal_px) / float(bbox_width)

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

        _, frame_width = frame.shape[:2]
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
            width = right - left
            center_x = left + width // 2
            center_y = top + (bottom - top) // 2
            distance_m = estimate_depth_m(width)
            distance_label = f"{distance_m:.2f}m" if distance_m is not None else "N/A"
            position_text = f"X: {center_x}px Y: {center_y}px Z: {distance_label}"
            font = cv2.FONT_HERSHEY_DUPLEX
            name_scale = 0.8
            info_scale = 0.6
            thickness = 1
            name_size, _ = cv2.getTextSize(name, font, name_scale, thickness)
            info_size, _ = cv2.getTextSize(position_text, font, info_scale, thickness)
            label_padding = 8
            line_spacing = 6
            info_box_height = name_size[1] + info_size[1] + (label_padding * 2) + line_spacing
            label_top = max(bottom - info_box_height, 0)
            label_width = max(right - left, name_size[0], info_size[0]) + (label_padding * 2)
            label_left = max(left - label_padding, 0)
            label_right = label_left + label_width
            if label_right > frame_width:
                label_right = frame_width
                label_left = max(label_right - label_width, 0)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(
                frame,
                (int(label_left), int(label_top)),
                (int(label_right), int(bottom)),
                color,
                cv2.FILLED,
            )
            text_x = int(label_left + label_padding)
            name_baseline_y = int(label_top + label_padding + name_size[1])
            info_baseline_y = int(name_baseline_y + line_spacing + info_size[1])
            cv2.putText(frame, name, (text_x, name_baseline_y), font, name_scale, (255, 255, 255), thickness)
            cv2.putText(frame, position_text, (text_x, info_baseline_y), font, info_scale, (255, 255, 255), thickness)

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
