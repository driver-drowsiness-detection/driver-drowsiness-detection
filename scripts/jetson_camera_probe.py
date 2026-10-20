"""Standalone Python 3.6+ CSI probe; imports neither MARVEL nor MediaPipe.

Run in the Jetson desktop after stopping the native camera preview.
"""
import argparse
import multiprocessing as mp
import queue
import re
import time

PIPELINE = ("nvarguscamerasrc sensor-id=0 ! nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink max-buffers=1 drop=true sync=false")


def capture_worker(pipeline, output, stop, errors):
    cap = None
    try:
        import cv2
        if not re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()):
            raise RuntimeError("Installed OpenCV has no GStreamer support")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError("Could not open CSI pipeline")
        index = 0
        while not stop.is_set():
            ok, image = cap.read()
            observed = time.monotonic()
            if not ok or image is None or image.size == 0:
                raise RuntimeError("CSI read failed")
            try:
                output.put_nowait((index, observed, image))
            except queue.Full:
                try:
                    output.get_nowait()
                except queue.Empty:
                    pass
            index += 1
    except Exception as exc:
        errors.send(str(exc))
    finally:
        if cap is not None:
            cap.release()
        errors.close()
        output.cancel_join_thread()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", default=PIPELINE)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if not 0 < args.timeout < float("inf") or args.max_frames < 0:
        parser.error("timeout must be finite and positive; max-frames must be nonnegative")
    import cv2
    import numpy as np

    print("OpenCV", cv2.__version__, cv2.__file__)
    print("Pipeline:", args.pipeline)
    context = mp.get_context("spawn")
    output, stop = context.Queue(1), context.Event()
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=capture_worker, args=(args.pipeline, output, stop, sender))
    worker.daemon = True
    worker.start()
    sender.close()
    last = time.monotonic()
    last_print, count = 0, 0
    try:
        while True:
            if receiver.poll():
                try:
                    message = receiver.recv()
                except EOFError:
                    message = "Camera worker exited"
                raise RuntimeError(message)
            try:
                index, observed, image = output.get(timeout=.1)
                last = time.monotonic()
                count += 1
                if last - last_print >= 1:
                    print("frame={} shape={} read_age={:.3f}s".format(index, image.shape, last-observed))
                    last_print = last
                if image.ndim != 3 or image.shape[2] != 3:
                    raise RuntimeError("Expected a BGR image with three channels")
                if args.max_frames and count >= args.max_frames:
                    break
            except queue.Empty:
                if not worker.is_alive():
                    raise RuntimeError("Camera worker exited")
                if time.monotonic() - last > args.timeout:
                    raise RuntimeError("No new CSI frame within timeout")
                image = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(image, "Waiting for CSI camera; q=quit", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            if not args.no_display:
                cv2.imshow("Jetson camera probe", image)
                if cv2.waitKey(1) & 0xff in (ord("q"), 27):
                    break
    finally:
        stop.set()
        worker.join(1)
        if worker.is_alive():
            worker.terminate()
            worker.join(1)
        output.close()
        receiver.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
