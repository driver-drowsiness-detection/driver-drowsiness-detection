def show(image, eyes, results, selected):
    import cv2
    view = image.copy()
    for eye in eyes:
        for x, y in eye.points:
            cv2.circle(view, (int(x), int(y)), 2, (0, 200, 0), -1)
    for line, (mode, result) in enumerate(results.items(), start=1):
        text = f"{mode}: {result['state']} active={result['alarm_active']}"
        cv2.putText(view, text, (10, 25*line), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if mode == selected else (255, 255, 255), 1)
    cv2.imshow("MARVEL", view)
    return cv2.waitKey(1) & 0xFF != ord("q")
