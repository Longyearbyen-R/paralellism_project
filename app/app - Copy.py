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
    def __init__(self, min_area_threshold=100):
        super().__init__()
        customtkinter.set_default_color_theme("dark-blue")
        self.geometry("500x400")
        self.title("Parallelism project")

        self.button = customtkinter.CTkButton(self, text="Choose a file", command=self.chooseFileCallback)
        self.button.pack(padx=20, pady=20)
        self.min_area_threshold = min_area_threshold
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    def chooseFileCallback(self):
        filename = 'C:/Users/throu/Downloads/Presence.jpg' # customtkinter.filedialog.askopenfilename()
        print(filename)
        df = self.scan_file(filename, 400)
        df.fillna("", inplace=True)
        print(tabulate(df, headers="keys", tablefmt="psql"))

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 150)
        return edges

    def find_table_region(self, edges, title_height):
        # Locate the position of the title in the image
        # You may need to customize this part based on the specific characteristics of your images
        # For demonstration, let's assume we already have the position of the title
        title_x, title_y = 100, 100  # Example position of the title
        
        # Define the region of interest (ROI) for the table
        table_roi = edges[title_y + title_height + 10:, :]

        return table_roi

    def extract_table(self, table_roi):
        # Use OCR to extract text from the table region
        table_text = pytesseract.image_to_string(table_roi)
        return table_text

    def process_table_text(self, table_text):
        # Define a regular expression pattern to match signatures
        signature_pattern = re.compile(r'\b(?:true|false)\b', re.IGNORECASE)
    
        # Split the text into lines
        lines = table_text.strip().split('\n')
    
        # Extract data rows
        data = []
        for line in lines:
            # Split the line into cells based on visible borders
            cells = line.split('|')
            
            # Remove empty cells and strip whitespace
            cells = [cell.strip() if cell.strip() else np.nan for cell in cells]
    
            # Fill missing cells with "false" for signature
            while len(cells) < 3:
                cells.append(np.nan)
    
            # Replace signature text with "signature"
            for i, cell in enumerate(cells):
                if signature_pattern.match(str(cell)):
                    cells[i] = "signature"
    
            data.append(cells)
    
        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=["Name", "Signature1", "Signature2"])
        return df

    def scan_file(self, file_path, title_height):
        # Read the image
        image = cv2.imread(file_path)

        # Preprocess the image
        edges = self.preprocess_image(image)

        # Find the region containing the table
        table_roi = self.find_table_region(edges, title_height)

        # Extract table text using OCR
        table_text = self.extract_table(table_roi)

        # Process table text and convert to DataFrame
        table_df = self.process_table_text(table_text)

        return table_df
