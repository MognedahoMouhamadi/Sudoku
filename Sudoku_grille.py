import cv2 as cv
import numpy as np

# Charger l'image en niveaux de gris
image = cv.imread("Sudoku_2_1.jpg")

# Vérifier que l'image a bien été chargée
if image is None:
    print("Erreur : Impossible de charger l'image.")
    exit()

# Convertir en niveaux de gris
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# Appliquer un seuillage adaptatif pour améliorer la distinction des bords
gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv.THRESH_BINARY, 11, 2)
# Inverser les couleurs pour que la grille soit blanche sur fond noir
gray = cv.bitwise_not(gray)

# Trouver les contours
contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Vérifier qu'on a trouvé des contours
if not contours:
    print("Aucun contour détecté.")
    exit()

# Trier les contours par aire décroissante et garder le plus grand
largest_contour = max(contours, key=cv.contourArea)

# Approximer le contour par un polygone à 4 sommets
epsilon = 0.02 * cv.arcLength(largest_contour, True)  # Facteur ajustable
approx = cv.approxPolyDP(largest_contour, epsilon, True)

# Vérifier si on a bien 4 sommets (une forme rectangulaire)
if len(approx) == 4:
    print("Grille détectée !")
else:
    print("Problème dans la détection.")

# Dessiner le plus grand contour sur une copie de l'image
#image_copie = image.copy()
#cv.drawContours(image_copie, [largest_contour], -1, (0, 0, 255), 3)

image_copie = image.copy()
cv.drawContours(image_copie, [largest_contour], -1, (0, 0, 255), 3)

# Afficher l'image avec le contour détecté
cv.imshow("Plus grand contour", image_copie)
cv.waitKey(0)
cv.destroyAllWindows()








