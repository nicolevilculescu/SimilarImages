# Write an application that uses a dataset of images (of at least) 30-50 images downloaded from internet with various content (landscapes, objects, people, etc). The application should take a new image as input argument and determine the set of similar images from the database:
#
# 1.       using all histogram comparison metrics available in OpenCV.
#
# 2.       By using a color reduction mechanism to transform similar colors from an equivalence class
# to its representative color and conducting the histogram comparison on the reduced color spaces
#
# Compare the results and explain.

import glob
import cv2
import matplotlib.pyplot as plt

index = {}
images = {}

other_image = cv2.imread('C:/Master/CV/labs/lab4/archive(1)/00000008_(5).jpg')
img = cv2.cvtColor(other_image, cv2.COLOR_BGR2RGB)
hist_image = cv2.calcHist([other_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
hist_image = cv2.normalize(hist_image, hist_image).flatten()

for i, imagePath in enumerate(glob.glob('C:/Master/CV/labs/lab4/archive(1)/photos_no_class' + '/*.jpg')):
    filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    images[i] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    index[i] = hist

# print(len(images))
# print(len(index))

OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),
    ("Chi Square", cv2.HISTCMP_CHISQR_ALT),
    ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA))

for (methodName, method) in OPENCV_METHODS:
    results = {}
    reverse = False

    if methodName in ("Correlation", "Intersection"):
        reverse = True

    for (k, hist) in index.items():
        d = cv2.compareHist(hist_image, hist, method)
        results[k] = d

    results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
    results = results[0:2]
    fig = plt.figure("Query")
    plt.imshow(other_image)
    plt.axis("off")
    fig = plt.figure("Results: %s" % methodName, figsize=(10, 10))
    fig.suptitle(methodName, fontsize=25)

    for (i, (v, k)) in enumerate(results):
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("%s:       %f" % (k, v))
        plt.imshow(images[k])
        plt.axis("off")

plt.show()