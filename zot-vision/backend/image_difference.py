import cv2
class ImageDifference:
    def __init__(self, threshold=30, min_changed_pixels=0.1):
        self.prev_frame = None
        self.threshold = threshold
        self.min_changed_pixels = min_changed_pixels

    def reset(self):
        self.prev_frame = None

    def detect(self, frame, name):
        if frame is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        has_changed = False
        if self.prev_frame is not None and self.prev_frame.shape == gray.shape:
            diff = cv2.absdiff(gray, self.prev_frame)
            changed = (diff > self.threshold).sum() / diff.size

            if changed > self.min_changed_pixels:
                cv2.imwrite(f"assets/{name}.jpg", frame)
                has_changed = True

        self.prev_frame = gray
        return has_changed
