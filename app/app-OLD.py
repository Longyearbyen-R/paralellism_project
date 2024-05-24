import customtkinter

from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from pytesseract import Output
from tabulate import tabulate
import pandas as pd
import numpy as np
import pytesseract
import argparse
import imutils
import cv2
import re

class App(customtkinter.CTk):
    def __init__(self, image_path='C:/Users/throu/Downloads/Presence.jpg',min_area_threshold=100):
        super().__init__()
        customtkinter.set_default_color_theme("dark-blue")
        self.geometry("500x400")
        self.title("Parallelism project")

        self.button = customtkinter.CTkButton(self, text="Choose a file", command=self.chooseFileCallback)
        self.button.pack(padx=20, pady=20)
        self.min_area_threshold = min_area_threshold
        self.image = cv2.imread(image_path)
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    def chooseFileCallback(self):
        filename = 'C:/Users/throu/Downloads/Presence.jpg' # customtkinter.filedialog.askopenfilename()
        print(filename)
        self.image_path = filename
        df = self.extract_table_data()
        #df.fillna("", inplace=True)
        
        print(tabulate(df, headers="keys", tablefmt="psql"))

    def locate_table(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h
    
    def extract_table_cells(self, table_region):
        x, y, w, h = table_region
        table_image = self.image[y:y+h, x:x+w]
        cell_images = []
        cell_height = h // 21
        cell_width = w // 3
        
        for row in range(21):
            for col in range(3):
                cell_x = x + col * cell_width
                cell_y = y + row * cell_height
                cell = self.image[cell_y:cell_y+cell_height, cell_x:cell_x+cell_width]
                cell_images.append(cell)
        
        return cell_images
    
    def analyze_cells(self, table_cells):
        data = []
        for cell in table_cells:
            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            binary_cell = cv2.adaptiveThreshold(gray_cell, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Check if the cell contains any black content
            has_black_content = cv2.countNonZero(binary_cell) > 0
            data.append(has_black_content)
        
        data = [data[i:i+3] for i in range(0, len(data), 3)]
        return data
    
    def extract_table_data(self):
        table_region = self.locate_table()
        table_cells = self.extract_table_cells(table_region)
        table_data = self.analyze_cells(table_cells)
        df = pd.DataFrame(table_data, columns=['Column1', 'Column2', 'Column3'])
        return df



"""
    def scanFile(self, filePath):
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        img = cv2.imread(filePath)

        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        print(d.keys())
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 60:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        imS = cv2.resize(img, (720, 1080)) 
        cv2.imshow('img', imS)
        cv2.waitKey(0)
"""