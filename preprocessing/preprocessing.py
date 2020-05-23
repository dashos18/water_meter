import cv2
import numpy as np
from scipy.optimize import minimize_scalar


class ImageProprocessor:

    @staticmethod
    def rotate_vertical_image(image):
        if image.shape[0] > image.shape[1]:
            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            new_image = cv2.warpAffine(image, M, (nW, nH))
        else:
            new_image = image
        return new_image

    @staticmethod
    def orient_numbers(image):
        img_r = ImageProprocessor.rotate_vertical_image(image)
        img = cv2.cvtColor(img_r, cv2.COLOR_BGR2HSV)
        boundaries = [([129, 75, 99], [179, 255, 255])]  # [169,100,100],[189,255,255]

        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv2.inRange(img, lower, upper)
            output = cv2.bitwise_and(img, img, mask=mask)
            #cv2.imshow('Only red',np.hstack([img,output]))
            y_start = 0
            y_end = output.shape[0]
            x_start = round(output.shape[1] / 2)
            x_end = output.shape[1]

            half_img = output[y_start:y_end, x_start:x_end]
            nonzero = np.argwhere(half_img != 0)
            # Just for now we can write that if both sides are black that we assume that they are both not rota
            if len(nonzero) == 0:
                (h, w) = img_r.shape[:2]
                (cX, cY) = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))
                M[0, 2] += (nW / 2) - cX
                M[1, 2] += (nH / 2) - cY
                new_image = cv2.warpAffine(img_r, M, (nW, nH))
            else:
                new_image = img_r

        return new_image


class CropperRotater:

    @staticmethod
    def crop_size_by_skew(w, h, alpha):
        alpha = np.deg2rad(alpha)
        x = np.array([[0, 0], [w, 0], [0, h]])
        t = 0.5 * np.array([[w, h], [w, h], [w, h]])
        x = x - t
        M = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        x = np.dot(x, M)
        x = x + t
        y1 = (x[1, 1] * x[0, 0] - x[0, 1] * x[1, 0]) / (x[0, 0] - x[1, 0])
        y2 = w * (x[0, 1] - x[1, 1]) / (x[0, 0] - x[1, 0]) + (x[1, 1] * x[0, 0] - x[0, 1] * x[1, 0]) / (
                    x[0, 0] - x[1, 0])
        x1 = (x[2, 0] * x[0, 1] - x[0, 0] * x[2, 1]) / (x[0, 1] - x[2, 1])
        x2 = h * (x[0, 0] - x[2, 0]) / (x[0, 1] - x[2, 1]) + (x[2, 0] * x[0, 1] - x[0, 0] * x[2, 1]) / (
                    x[0, 1] - x[2, 1])
        x_crop = int(max(x1, x2))
        y_crop = int(max(y1, y2))
        return (x_crop, y_crop, w - 2 * x_crop, h - 2 * y_crop)

    @staticmethod
    def deskew(image):
        def minimize_func_x(angle):
            rows, cols = sobelx.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            dst = cv2.warpAffine(sobelx, M, (cols, rows), flags=cv2.INTER_LANCZOS4)
            proj = np.sum(dst, axis=0)
            out = np.std(proj)
            return -out

        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.normalize(im_gray, im_gray, 0, 255, cv2.NORM_MINMAX)
        img = cv2.GaussianBlur(im_gray, (3, 3), 0)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        h, w = im_gray.shape
        mask = np.zeros((h, w), dtype=float)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2 - 5, w // 2 - 5), 0, 0, 360, 1, -1)
        sobelx = sobelx * mask

        res_x = minimize_scalar(minimize_func_x, bounds=(-9, 9), method='golden', tol=0.03)
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), res_x.x, 1)
        dst = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LANCZOS4)
        x, y, w, h = CropperRotater.crop_size_by_skew(cols, rows, res_x.x)
        # print('r ',rows, 'c ',cols)
        # print('xywh-',x,y,w,h)

        return dst[y:y + h, x:x + w]

    @staticmethod
    def find_color_mask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        h_mask1 = cv2.threshold(h, .8 * 180, 255, cv2.THRESH_BINARY)[1]
        h_mask2 = cv2.threshold(h, .02 * 180, 255, cv2.THRESH_BINARY_INV)[1]
        h_mask = cv2.bitwise_or(h_mask1, h_mask2)
        s_mask = cv2.threshold(s, 0.15 * 255, 255, cv2.THRESH_BINARY)[1]
        v_mask1 = cv2.threshold(v, .8 * 255, 255, cv2.THRESH_BINARY_INV)[1]
        v_mask2 = cv2.threshold(v, .2 * 255, 255, cv2.THRESH_BINARY)[1]
        v_mask = cv2.bitwise_and(v_mask1, v_mask2)

        mask = cv2.bitwise_and(v_mask, h_mask)
        mask = cv2.bitwise_and(mask, s_mask)
        mask[:, 0:mask.shape[1] // 2] = 0  # delete noise at left side
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)

        return mask

    @staticmethod
    def finalize(image):
        dst = CropperRotater.deskew(image)
        return dst
