""" Simple blob detection implementation using OpenCV """

import cv2
import numpy as np

def setup_params():
    """ setup detector with given parameters """

    # creates detector with default parameters
    params = cv2.SimpleBlobDetector_Params()

    # threshold params
    params.minThreshold = 0
    params.maxThreshold = 255

    # area params
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 10000

    # color params
    params.filterByColor = False
    params.blobColor = 0
    params.minCircularity = 0.5
    params.maxCircularity = 1

    # convexity params
    params.filterByConvexity = False
    params.minConvexity = 0.5
    params.maxConvexity = 1
    params.minDistBetweenBlobs = 0

    return cv2.SimpleBlobDetector_create(params)

def find_blobs(img):
    """ finds blobs in given thermal image"""

    # load input image
    original_img = cv2.imread(img, 0)

    # creates detector
    detector = setup_params()

    # detect blobs in image
    keypoint_info = detector.detect(original_img)

    # highlight blobs as circles
    blobs = cv2.drawKeypoints(original_img, keypoint_info, np.array([]),
                                (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # display image w/ blobs
    cv2.imshow("[!] Displaying Blobs [!]", blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# driver code
if __name__ == '__main__':
    find_blobs("images/test1.jpg")    # mug
    find_blobs("images/test2.jpg")    # person
