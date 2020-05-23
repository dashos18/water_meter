import cv2
import numpy as np

class NeuralNet:
    def __init__(
            self,
            confidence_thresh,
            NMS_thresh,
            network_resolution,
            path_config,
            path_weights
    ):
        self.confidence_thresh = confidence_thresh
        self.NMS_thresh = NMS_thresh
        self.input_width, self.input_height = network_resolution, network_resolution
        self.config = path_config
        self.weights = path_weights
        self.net = self.setup_net()

    def setup_net(self):
        neural_net = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return neural_net

    def create_blob(self, image):
        """
       Creates a blob out of the image provided. Returns the blob
       """
        try:
            blob = cv2.dnn.blobFromImage(
                image, 1 / 255,
                (self.input_width, self.input_height),
                [0, 0, 0], 1, crop=False
            )
        except Exception as e:
            print(f"Failed while creating the blob. Error: {e}")
            raise

        return blob

    def output_layers(self, net):
        """
        Returns names of the output YOLO layers: ['yolo_82', 'yolo_94', 'yolo_106']
        """
        layers = net.getLayerNames()

        return [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def process_predictions(self, image, predictions):
        """
        Process all BBs predicted. Keep only the valid ones.
        """
        image_height, image_width = image.shape[0], image.shape[1]
        classIds, confidences, boxes = [], [], []
        objects_predicted = list()

        # For each prediction from each of 3 YOLO layers
        for prediction in predictions:
            # For each detection from one YOLO layer
            for detection in prediction:
                scores = detection[5:]
                classId = np.argmax(scores)  # Index of a BB with highest confidence
                confidence = scores[classId]  # Value of this BB's confidence

                if confidence > self.confidence_thresh:
                    # Centre of object relatively to the upper left corner in percent
                    centre_x = int(detection[0] * image_width)
                    centre_y = int(detection[1] * image_height)

                    # Width and height of the BB predicted. Check for ERROR
                    width_percent = detection[2] if detection[2] < 0.98 else 0.98
                    height_percent = detection[3] if detection[3] < 0.98 else 0.98

                    # Calculate actual size of the BB
                    width = int(width_percent * image_width)
                    height = int(height_percent * image_height)

                    # ERROR CATCHING WITH ABS
                    left = int(centre_x - (width / 2)) if int(centre_x - (width / 2)) > 0 else 2
                    top = int(centre_y - (height / 2)) if int(centre_y - (height / 2)) > 0 else 2

                    # Save prediction results
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non-max suppression to eliminate redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_thresh, self.NMS_thresh)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            objects_predicted.append([classIds[i], confidences[i], left,
                                      top, left + width, top + height])

        # print('This is object predicted:',objects_predicted )
        return objects_predicted

    def predict(self, image: np.ndarray) -> list:
        """
        Performs serial number and counter detection and classification. Returns list of objects detected
        """
        blob = self.create_blob(image)
        # Pass the blob to the neural net
        self.net.setInput(blob)
        # Get output YOLO layers from which read predictions
        layers = self.output_layers(self.net)
        # Run forward pass and get predictions from 3 YOLO layers
        predictions = self.net.forward(layers)
        # Parse the predictions, save only the valid ones
        components = self.process_predictions(image, predictions)

        return components