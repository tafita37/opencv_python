import cv2
import numpy as np

# Charger l'image dentaire
image = cv2.imread('sourire.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer un seuillage pour obtenir une image binaire
_, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Trouver les contours dans l'image
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dessiner les contours sur l'image originale
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Afficher l'image originale avec les contours
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
