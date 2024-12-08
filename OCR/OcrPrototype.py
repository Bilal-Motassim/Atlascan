import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import os

#Amelioration de la visibilité de l'image pour le model Paddle
def enhance_image_for_ocr(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresholded > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = thresholded.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresholded, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    resized = cv2.resize(rotated, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    enhanced_image_path = 'enhanced_image.jpg'
    cv2.imwrite(enhanced_image_path, resized)

    return enhanced_image_path

def extract_text_from_image(image_path, output_file_path):
    enhanced_image_path = enhance_image_for_ocr(image_path)

    ocr = PaddleOCR(lang='ar')  

    result = ocr.ocr(enhanced_image_path)

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write("=== Extracted Text ===\n")
        
        for line in result[0]:
            text = line[1][0]  
            output_file.write(f"{text}\n")

    print(f"Extraction complete! Results saved to: {output_file_path}")


document_path = r"C:\Users\pc\Desktop\TempFiles\CIN_maroc.jpg"
output_file_path = r"C:\Users\pc\Desktop\TempFiles"  # Output file path

extract_text_from_image(document_path, output_file_path)
