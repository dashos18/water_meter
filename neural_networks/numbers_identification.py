class NumberDetector:
    def __init__(self, net):
        self.net = net
        print("Numbers detector initialized")

    def detect_numbers(self, image):
        predictions = self.net.predict(image)
        return predictions
