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
import concurrent.futures
import threading
import queue
import time

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

        self.result_queue = queue.Queue()

    def chooseFileCallback(self):
        filenames = customtkinter.filedialog.askopenfilename(multiple = True)
        print(filenames)
        self.scanAllFiles_MultiThreads_Producteur_Consommateur(list(filenames))

    def process_file(self, filename):
        try:
            df = self.scan_file(filename, 400)
            df.fillna("", inplace=True)
            self.result_queue.put(df)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    def scanAllFiles_MultiThreads(self, filenames):
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_file, filenames))

        for df in results:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

    def scanAllFiles_MultiProcessus(self, filenames):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(process_file, [self] * len(filenames), filenames))

        for df in results:
            print(tabulate(df, headers="keys", tablefmt="psql"))

    def process_file(self, filename):
        df = self.scan_file(filename, 400)
        df.fillna("", inplace=True)
        return df

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 150)
        return edges

    def find_table_region(self, edges, title_height):
        title_x, title_y = 100, 100
        
        table_roi = edges[title_y + title_height + 10:, :]

        return table_roi

    def extract_table(self, table_roi):
        table_text = pytesseract.image_to_string(table_roi)
        return table_text

    def process_table_text(self, table_text):
        signature_pattern = re.compile(r'\b(?:true|false)\b', re.IGNORECASE)
        lines = table_text.strip().split('\n')
        data = []
        for line in lines:
            cells = line.split('|')
            cells = [cell.strip() if cell.strip() else np.nan for cell in cells]
            while len(cells) < 3:
                cells.append(np.nan)

            for i, cell in enumerate(cells):
                if signature_pattern.match(str(cell)):
                    cells[i] = "signature"
    
            data.append(cells)
        df = pd.DataFrame(data, columns=["Name", "Signature1", "Signature2"])
        return df

    def scan_file(self, file_path, title_height):
        image = cv2.imread(file_path)
        edges = self.preprocess_image(image)
        table_roi = self.find_table_region(edges, title_height)
        table_text = self.extract_table(table_roi)
        table_df = self.process_table_text(table_text)
        return table_df

    def producer(self, filenames):
        for filename in filenames:
            df = self.scan_file(filename, 400)
            self.result_queue.put(df)

    def consumer(self):
        while True:
            df = self.result_queue.get()
            if df is None:
                break
            df.fillna("", inplace=True)
            print(tabulate(df, headers="keys", tablefmt="psql"))
            self.result_queue.task_done()

    def scanAllFiles_MultiThreads_Producteur_Consommateur(self, filenames):
        start_time = time.time()

        producer_thread = threading.Thread(target=self.producer, args=(filenames,))
        producer_thread.start()

        consumer_threads = []
        for _ in range(len(filenames)):
            consumer_thread = threading.Thread(target=self.consumer)
            consumer_thread.start()
            consumer_threads.append(consumer_thread)

        producer_thread.join()

        for _ in consumer_threads:
            self.result_queue.put(None)

        for consumer_thread in consumer_threads:
            consumer_thread.join()

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

"""

    def scanAllFiles_MultiThreads(self, filenames):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for filename in filenames:
                executor.submit(self.process_file, filename)

        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())

        for df in results:
            print(tabulate(df, headers="keys", tablefmt="psql"))
"""
