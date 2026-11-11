import importlib
import json
import platform
import shutil
import sys


def main():
    result = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "machine": platform.machine(), "modules": {}}
    for name in ("numpy", "PIL", "yaml", "cv2", "mediapipe", "torch", "onnx", "onnxruntime"):
        try:
            module = importlib.import_module(name)
            result["modules"][name] = {"version": getattr(module, "__version__", "unknown"),
                                       "file": getattr(module, "__file__", None)}
            if name == "cv2":
                result["opencv_build"] = module.getBuildInformation()
            if name == "torch":
                result["cuda_available"] = module.cuda.is_available()
                result["torch_cuda"] = module.version.cuda
        except Exception as exc:
            result["modules"][name] = {"error": str(exc)}
    result["executables"] = {x: shutil.which(x) for x in ("gst-launch-1.0", "aplay", "espeak-ng", "llama-server")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
