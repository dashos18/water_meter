from neural_networks.neural_netork import NeuralNet
from neural_networks.counter_identification import WaterCounterDetector
from neural_networks.numbers_identification import  NumberDetector
from preprocessing.preprocessing import ImageProprocessor, CropperRotater
from utils.drawer import Drawer
import os
import cv2


def main():

    test_dir = '/Users/dariavolkova/Desktop/TESTING_ENTIRE_ALGORTHIM/200_images'
    # Initialize neural nets
    counters_net = NeuralNet(
        confidence_thresh=0.2,
        NMS_thresh=0.5,
        network_resolution=608,
        path_config='/Users/dariavolkova/PycharmProjects/Numbers_detection/dependencies/yolo_meters.cfg',
        path_weights='/Users/dariavolkova/PycharmProjects/Numbers_detection/dependencies/yolo_meters.weights'
    )
    numbers_net = NeuralNet(
        confidence_thresh=0.2,
        NMS_thresh=0.5,
        network_resolution=800,
        #network_resolution=608
        path_config='/Users/dariavolkova/PycharmProjects/Numbers_detection/dependencies/yolo_numbers.cfg',
        #path_weights='/Users/dariavolkova/PycharmProjects/Numbers_detection/dependencies/yolo_numbers.weights'
        path_weights='/Users/dariavolkova/PycharmProjects/Numbers_detection/dependencies/yolo-obj_final_numbers.weights'
    )
    # Initialize detectors
    counter_detector = WaterCounterDetector(net=counters_net)
    numbers_detector = NumberDetector(net=numbers_net)

    # Initialize preprocessing
    image_preprocessor = ImageProprocessor()
    cropper_rotater = CropperRotater()

    for file in os.listdir(test_dir):
        path_to_file = os.path.join(test_dir, file)

        if not any(file.endswith(ext.lower()) for ext in [".jpg", ".jpeg", ".png"]):
            continue
        try:
            image = cv2.imread(path_to_file)
        except Exception as e:
            print(f"Failed during decoding an image: {file}. Error: {e}")
            continue

        # STEP 1 - detect water meter on the image
        counter = counter_detector.detect_meter(image)
        if len(counter) == 0:
            print(f" This file just failed: {file}")
            continue

        # STEP 2 - preprocess images by rotating them
        rotated_meter = image_preprocessor.orient_numbers(counter)
        processed_meter = cropper_rotater.finalize(rotated_meter)

        # STE3 3 - predict numbers
        number_predictions = numbers_detector.detect_numbers(processed_meter)

        # STEP 4 - post process number predictions, draw BB, write classes
        #image_out = Drawer.visualise_results(number_predictions, processed_meter)

        #STEP 5 - show image and predicted numbers
        recoognized_numbers = list()
        predictions_sorted = sorted(number_predictions, key=lambda x: x[2])
        for numbers in predictions_sorted:
            recoognized_numbers.append(numbers[0])
        recoognized_numbers.insert(5, '.')
        recoognized_numbers = ''.join([str(i) for i in recoognized_numbers])
        cv2.putText(image, 'Predicted numbers: ' + str(recoognized_numbers) + 'kL', (25, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print('This is recognized numbers', recoognized_numbers)
        cv2.imshow('', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()