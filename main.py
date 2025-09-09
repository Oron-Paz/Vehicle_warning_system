import cv2 as cv
import numpy as np
import pandas as pd
import os

DATA_FOLDER_PATH= "car_data/data/export/"
df = pd.read_csv(DATA_FOLDER_PATH + "_annotations.csv")

print("Column names:", df.columns.tolist())
print("First few rows:")
print(df.head())
print(f"Total rows: {len(df)}")



#img = cv.imread(DATA_FOLDER_PATH + "1478019952686311006_jpg.rf.54e2d12dbabc46be3c78995b6eaf3fee.jpg", cv.IMREAD_COLOR)
#cv.imshow('Original Image', img)
#cv.waitKey(0)
#cv.destroyAllWindows()

