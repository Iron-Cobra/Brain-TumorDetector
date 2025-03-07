import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import ctypes
import numpy as np
from keras.models import load_model
import ttkbootstrap as ttk

# Enable high DPI awareness for Windows to prevent blurry text
ctypes.windll.shcore.SetProcessDpiAwareness(1)

# Load the model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Create main window
root = ttk.Window(themename="superhero")
root.call("tk", "scaling", 1.5)
root.title("AI-Powered Tumor Detection")
root.geometry("1280x720")
root.configure(bg="#1C1E26")
root.resizable(True, True)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

def process_image(file_path):
    processed_img = preprocess_image(file_path)
    prediction = model.predict(processed_img)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index] * 100  # Convert to percentage

    # Determine result message and color
    if "no" in class_name.lower():
        result_text = f"✅ No Tumor Detected.\nConfidence Score: {confidence_score:.2f}%"
        result_label.config(text=result_text, foreground="#00C853")
    else:
        result_text = f"⚠️ Tumor Detected!\nConfidence Score: {confidence_score:.2f}%"
        result_label.config(text=result_text, foreground="#E74C3C")

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    if file_path:
        img = Image.open(file_path).convert("RGB")
        original_width, original_height = img.size
        max_size = 500
        
        if original_width > original_height:
            new_width = max_size
            new_height = int((max_size / original_width) * original_height)
        else:
            new_height = max_size
            new_width = int((max_size / original_height) * original_width)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        
        process_image(file_path)

# Styling
header_label = ttk.Label(root, text="AI-Powered Tumor Detection", font=("Arial", 24, "bold"), foreground="white", background="#1C1E26")
header_label.pack(pady=30)

upload_button = ttk.Button(root, text="📂 Select Image", command=upload_image, bootstyle="primary-outline")
upload_button.pack(pady=20)

frame = ttk.Frame(root, bootstyle="dark")
frame.pack(pady=30)
image_label = ttk.Label(frame)
image_label.pack()

result_label = ttk.Label(root, text="", font=("Arial", 18, "bold"), foreground="#E74C3C", background="#1C1E26")
result_label.pack(pady=20)

root.mainloop()
