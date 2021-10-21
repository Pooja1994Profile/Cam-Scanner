from flask import Flask, render_template, request, jsonify
from flask_cors import cross_origin
import cv2
import imutils
from skimage.filters import threshold_local
import numpy as np
import pytesseract


app = Flask(__name__)


@app.route('/', methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")


def order_points(pts):  # to find the endpoints of the square use below function
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def transformFourPoints(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    # Applying perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


@app.route('/scan', methods=['POST', 'GET'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            filePath = request.form['scanimg']

            # load the image and compute the ratio of the old height to the new height, clone it, and resize it
            image = cv2.imread(filePath)
            ratio = image.shape[0] / 500.0
            orig = image.copy()
            image = imutils.resize(image, height=500)

            # convert colored image to Gray Scale image, blur it, and find edges
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # Edge Detection
            edged = cv2.Canny(gray, 75, 200)

            cv2.imshow("Image", image)
            cv2.imshow("Edged", edged)
            # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Run loop for all the contours
            for c in contours:

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                if len(approx) == 4: # Check square or not, If return 4 points the square
                    target = approx
                    break

            # show the contour (outline) of the piece of paper
            cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
            cv2.imshow("Outline", image)

            # apply the four point transform to obtain a top-down
            warped = transformFourPoints(orig, target.reshape(4, 2) * ratio)

            # convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            T = threshold_local(warped, 11, offset=10, method="gaussian")
            warped = (warped > T).astype("uint8") * 255

            cv2.imshow("Original", imutils.resize(orig, height=650))
            cv2.imshow("Scanned", imutils.resize(warped, height=650))

            # Extract text from scanned image
            text = pytesseract.image_to_string(warped, lang='eng')

            return render_template('result.html', scantext=text)

        except Exception as e:
            print('The Exception message is: ', e)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)