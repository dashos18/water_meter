import cv2


class Drawer:

    @staticmethod
    def visualise_results(elements, image):

        for element in elements:
            frame_centre = (image.shape[1] // 2, image.shape[0] // 2)
            object_id = 0
            color = (204, 0, 102)
            # Convert element's coordinates to absolute (now they are relative to the
            # object within which they were detected (if any, else already absolute)
            element_absolute_left = element[2]
            element_absolute_top = element[3]
            element_absolute_right = element[4]
            element_absolute_bot = element[5]

            if element[0] == 0:
                cv2.rectangle(image,
                              (element_absolute_left, element_absolute_top),
                              (element_absolute_right, element_absolute_bot),
                              color, 2)
                cv2.putText(image, '0',
                            (element_absolute_left, element_absolute_top + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif element[0] == 1:
                cv2.rectangle(image,
                              (element_absolute_left, element_absolute_top),
                              (element_absolute_right, element_absolute_bot),
                              color, 2)
                cv2.putText(image, '1', (element_absolute_left, element_absolute_top + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif element[0] == 2:
                cv2.rectangle(image,
                              (element_absolute_left, element_absolute_top),
                              (element_absolute_right, element_absolute_bot),
                              color, 2)
                cv2.putText(image, '2',
                            (element_absolute_left, element_absolute_top + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif element[0] == 3:
                cv2.rectangle(image,
                              (element_absolute_left, element_absolute_top),
                              (element_absolute_right, element_absolute_bot),
                              color, 2)
                cv2.putText(image, '3',
                            (element_absolute_left, element_absolute_top + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif element[0] == 4:
                cv2.rectangle(image,
                              (element_absolute_left, element_absolute_top),
                              (element_absolute_right, element_absolute_bot),
                              color, 2)
                cv2.putText(image, '4',
                            (element_absolute_left, element_absolute_top + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif element[0] == 5:
                cv2.rectangle(image,
                              (element_absolute_left, element_absolute_top),
                              (element_absolute_right, element_absolute_bot),
                              color, 2)
                cv2.putText(image, '5',
                            (element_absolute_left, element_absolute_top + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif element[0] == 6:
                cv2.rectangle(image,
                              (element_absolute_left, element_absolute_top),
                              (element_absolute_right, element_absolute_bot),
                              color, 2)
                cv2.putText(image, '6',
                            (element_absolute_left, element_absolute_top + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif element[0] == 7:
                cv2.rectangle(image,
                              (element_absolute_left, element_absolute_top),
                              (element_absolute_right, element_absolute_bot),
                              color, 2)
                cv2.putText(image, '7',
                            (element_absolute_left, element_absolute_top + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif element[0] == 8:
                cv2.rectangle(image,
                              (element_absolute_left, element_absolute_top),
                              (element_absolute_right, element_absolute_bot),
                              color, 2)
                cv2.putText(image, '8',
                            (element_absolute_left, element_absolute_top + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif element[0] == 9:
                cv2.rectangle(image,
                              (element_absolute_left, element_absolute_top),
                              (element_absolute_right, element_absolute_bot),
                              color, 2)
                cv2.putText(image, '9',
                            (element_absolute_left, element_absolute_top + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return image