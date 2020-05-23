
class WaterCounterDetector:
    def __init__(self, net):
        # Initialize the net provided
        self.net = net
        print("Water counter detector initialized")

    def detect_meter(self, image):

        predictions = self.net.predict(image)
        original_image = image.copy()

        if predictions:
            for element in predictions:
                if element[0] == 0:
                    cropped = original_image[element[3]-15:element[5]+15, element[2]-15:element[4]+15]
                    return cropped

        return list()