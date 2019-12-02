import pytesseract
from pytesseract import Output
import cv2
import numpy as np

def roi (img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

pytesseract.pytesseract.tesseract_cmd = 'E:\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('images\\pantest3.jpg')
#resize 1.5858
dim = (600, 378)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
org = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.Canny(img, threshold1=200, threshold2=300)

vertices = np.array ([[8,87],[337,87],[337,278],[8,278]], np.int32)
img = roi(img, [vertices])

#rect = (8:337, 87:278)
#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
text = pytesseract.image_to_string(img)  
print(text)  
d = pytesseract.image_to_data(img, output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(org, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('output', org)
cv2.waitKey(0)
