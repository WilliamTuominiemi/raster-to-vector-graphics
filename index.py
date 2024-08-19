from PIL import Image
import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = Image.open("raster.jpg")
resized_image = img.resize((int(img.size[0]/4), int(img.size[1]/4))) # make the image more "icon" sized
open_cv_image = np.array(resized_image) # turn the image into a numpy array, which opencv understands
open_cv_image = open_cv_image[:, :, ::-1].copy() # convert from rgb to bgr (PIL and CV use different)
gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY) # grayscale image
retval, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # threshold (grayscale to binary)
contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # get all contours from image

# create svg file from the contours
with open("vector.svg", "w+") as f:
    f.write(f'<svg width="{resized_image.width}" height="{resized_image.height}" xmlns="http://www.w3.org/2000/svg">\n')
    for c in contours:

        # I had a bug where the whole image was seen as one big contour, which created a "frame"
        area = cv2.contourArea(c)
        if area > (resized_image.width - 10) * (resized_image.height - 10): # skip contours which closely match the image size
            continue

        f.write('<path d="M')

        x, y = c[0][0]
        f.write(f"{x} {y} ") # write first point

        for i in range(1, len(c)):
            x, y = c[i][0]
            f.write(f"L{x} {y} ") # write rest of the lines using "L"

        f.write('Z" style="stroke:black; fill:none;"/>\n') # close path if closed contour
    
    f.write("</svg>")