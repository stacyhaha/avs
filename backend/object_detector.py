from PIL import Image

PEDESTRIAN = "pedestrian"


class ObjectDetector:
    def __init__(self):
        return

    def detect(self, image: Image):
        """
        return the detect result and the drving command
        for example:
        (None, None, None, None): detect nothing, don't need to adjust driving
        ("pedestrian", 0, 0, 0): detect pedestrian, need to stop the car immediately
        ("pedestrian", None, None, None): detect pedestrain, don't need to take action
        """
        return (PEDESTRIAN, None, None, None)
