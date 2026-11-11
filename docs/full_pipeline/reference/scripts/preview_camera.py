"""python scripts/preview_camera.py configs/runtime.yaml"""
import sys
from pathlib import Path


def main():
    import cv2
    import yaml
    from marvel.perception.capture import LiveCapture, VideoCapture
    config = yaml.safe_load(Path(sys.argv[1]).read_text())["capture"]
    live = config["kind"] != "video"
    capture = LiveCapture(config) if live else VideoCapture(config)
    try:
        while True:
            frame = capture.read()
            if frame is None:
                if not live:
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            cv2.imshow("Camera preview", frame.image)
            print(frame.sample_id, round(frame.t, 3), frame.image.shape)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
