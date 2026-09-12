# Jetson environment record

Status: unverified. niharika526 records the runtime inventory; shamaa007 checks capture. Vishruth and Arushuuu assess detector and CNN compatibility and review this file.

Record board model, OS, Jetson Linux / JetPack, architecture, Python, RAM, camera connection, OpenCV build and available inference runtime. Preserve exact commands and errors. Do not replace system Python as a shortcut.

First board slot: inspect existing software, test camera separately, then test one candidate model on a saved input. Save measured latency and output parity when available.

Do not copy Windows dependency pins onto Jetson. Installation instructions will be added after compatibility is demonstrated.
