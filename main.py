import cv2 as cv
import pandas as pd
import os

DATA_FOLDER_PATH = "car_data/data/export"

df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, "_annotations.csv"))  # or whatever your CSV is named

# Use the first filename from the CSV instead of hardcoding
filename = df['filename'].iloc[0]
print(f"Loading: {filename}")

image = cv.imread(os.path.join(DATA_FOLDER_PATH, filename))

image_rows = df[df['filename'] == filename]
print(f"Found {len(image_rows)} annotations for this image:")
print(image_rows)

for index, row in image_rows.iterrows():
    xmin, xmax, ymin, ymax = row['xmin'], row['xmax'], row['ymin'], row['ymax']
    cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # green box (image, min_cord, max_cord, color, thickenss of border)
    cv.putText(image, row['class'],(row['xmin'], row['ymax']), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2) # rectangle title (image, title_text, pos_cord, font, size, color, thickness)
    

if image is None:
    print("Still can't load image - check if file exists")
else:
    cv.imshow('Original Image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()