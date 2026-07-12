"""YOLOv8 detection wrapper.

Expects a model fine-tuned on the Roboflow football-players-detection
dataset (classes: ball, goalkeeper, player, referee). A stock COCO model
will not work here -- it has no referee/goalkeeper classes and its
'sports ball' detection is unreliable at broadcast scale.
"""

from ultralytics import YOLO


class Detector:
    def __init__(self, model_path, conf=0.1, batch_size=16):
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.batch_size = batch_size
        self.class_names = self.model.names  # id -> name

    def detect(self, frames, progress=False):
        """Run detection over a list of BGR frames. Returns ultralytics Results."""
        results = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i : i + self.batch_size]
            results += self.model.predict(batch, conf=self.conf, verbose=False)
            if progress:
                print(f"  detection {min(i + self.batch_size, len(frames))}/{len(frames)} frames")
        return results
