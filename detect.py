
import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent

FIRE_SMOKE_MODEL = ROOT / "models" / "fire_smoke_best.pt"
PERSON_MODEL = "yolo11n.pt"  # COCO pretrained model

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------


def load_models(detect_person=False):
    if not FIRE_SMOKE_MODEL.exists():
        raise FileNotFoundError(
            f"Model not found: {FIRE_SMOKE_MODEL}"
        )

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Inference device:", device)

    fire_smoke = YOLO(str(FIRE_SMOKE_MODEL))

    person_model = None
    if detect_person:
        person_model = YOLO(PERSON_MODEL)

    print("Fire/Smoke classes:", fire_smoke.names)
    return fire_smoke, person_model, device


def detect(frame, fire_smoke, person_model, device, conf):
    # Fire and smoke
    result = fire_smoke.predict(
        source=frame,
        conf=conf,
        imgsz=640,
        device=device,
        verbose=False
    )[0]

    output = result.plot()

    # Optional person detection
    if person_model is not None:
        person_result = person_model.predict(
            source=frame,
            classes=[0],  # COCO person
            conf=conf,
            imgsz=640,
            device=device,
            verbose=False
        )[0]

        output = person_result.plot(img=output)

    return output


def test_image(args, fire_smoke, person_model, device):
    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Unable to read the image.")

    output = detect(
        image, fire_smoke, person_model, device, args.conf
    )

    save_path = OUTPUT_DIR / f"{image_path.stem}_detected.jpg"
    cv2.imwrite(str(save_path), output)

    print("Saved result:", save_path)

    cv2.imshow("Scrapyard Fire-Smoke Detection", output)
    print("Press any key to close the image.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_webcam(args, fire_smoke, person_model, device):
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}"
        )

    print("Webcam started. Press Q to quit.")

    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam frame.")
            break

        output = detect(
            frame, fire_smoke, person_model, device, args.conf
        )

        current_time = time.perf_counter()
        elapsed = current_time - prev_time
        prev_time = current_time

        fps = 1.0 / elapsed if elapsed > 0 else 0

        cv2.putText(
            output,
            f"Processing FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Scrapyard Drone AI - Webcam", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Scrapyard fire/smoke/person detection"
    )

    parser.add_argument(
        "--mode",
        choices=["image", "webcam"],
        required=True
    )

    parser.add_argument(
        "--image",
        default=str(ROOT / "images" / "test.jpg")
    )

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--person", action="store_true")

    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")

    fire_smoke, person_model, device = load_models(
        detect_person=args.person
    )

    if args.mode == "image":
        test_image(args, fire_smoke, person_model, device)
    else:
        test_webcam(args, fire_smoke, person_model, device)


if __name__ == "__main__":
    main()


# python detect.py --mode image
# python detect.py --mode image --image images/test.jpg
# python detect.py --mode webcam --person
