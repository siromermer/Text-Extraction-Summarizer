import tkinter as tk
from tkinter import filedialog, scrolledtext
import cv2
from PIL import Image, ImageTk
import pytesseract
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer

from googletrans import Translator
from tkinter import ttk
import time

MAX_WINDOW_WIDTH = 600
MAX_WINDOW_HEIGHT = 600
PADDING = 10

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# XSUM Model
tokenizer_xsum = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model_xsum = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

# Pegasus cnn_dailymail 
tokenizer_daily = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
model_daily = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")

# bart model 
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')



# List to store image paths
image_paths = []

def open_file():
    file_paths = filedialog.askopenfilenames(initialdir="./images", title="Select file",
                                             filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp"),
                                                        ("All files", "*.*")))
    if file_paths:
        for file_path in file_paths:
            image_paths.append(file_path)
            display_image(file_path)
        show_buttons()

def display_image(file_path):
    global thresh1,image
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    # Add padding to the binary image
    padded_image = cv2.copyMakeBorder(gray, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=0)
    
    # Calculate scale factor based on the maximum window size
    scale_factor = min(MAX_WINDOW_WIDTH / padded_image.shape[1], MAX_WINDOW_HEIGHT / padded_image.shape[0], 1.0)
    scale_factor=scale_factor-0.07
    # Resize image if it exceeds the maximum window size
    if scale_factor < 1.0:
        resized_image = cv2.resize(padded_image, (int(padded_image.shape[1] * scale_factor), int(padded_image.shape[0] * scale_factor)))
    else:
        resized_image = padded_image
    
    resized_image = Image.fromarray(resized_image)
    resized_image = ImageTk.PhotoImage(resized_image)

    # Determine the number of columns in row 1
    num_columns = root.grid_size()[0]

    if padded_image.shape[0] > padded_image.shape[1]:
        panel = tk.Label(root, image=resized_image)
        panel.image = resized_image
        panel.grid(row=1, column=1, columnspan=num_columns, padx=40)  # Span across all columns in row 1
    else:
        panel = tk.Label(root, image=resized_image)
        panel.image = resized_image
        panel.grid(row=1, column=1, columnspan=num_columns)

def show_buttons():
    text_button.grid(row=0, column=1, pady=10)

def extract_text():
    global combined_content

    combined_content = ""
    for file_path in image_paths:
        text = extract_text_from_image(file_path)
        combined_content += text.strip() + " "

    combined_content = "\n".join(line.strip() for line in combined_content.split("\n") if line.strip())

    text_window = tk.Toplevel(root)
    text_window.title("Extracted Text")
    text_window.geometry("800x600")

    # Create a Canvas widget
    canvas = tk.Canvas(text_window)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add a frame to the canvas
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor=tk.NW)

    text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
    text_area.insert(tk.END, "        Extracted Content \n")
    text_area.insert(tk.END, combined_content)
    text_area.pack(padx=75, pady=20, expand=True, fill='x')


    def summary_creator():

        selected_model=model_combobox.get()

        model_codes = {
            'xsum': [model_xsum,tokenizer_xsum],
            'pegasus-daily-maily': [model_daily,tokenizer_daily],
            'bart-daily-maily': [bart_model,bart_tokenizer],
            # Add more language codes as needed
        }

        model_list = model_codes.get(selected_model)

        model=model_list[0]
        tokenizer=model_list[1]
        
        tokens = tokenizer(combined_content, truncation=True, padding="longest", return_tensors="pt")
        summary = model.generate(**tokens, min_length=35)
        summary_text = tokenizer.decode(summary[0])
        summary_text = summary_text.replace("<pad>", "").replace("</s>", "")

        summary_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
        summary_widget.insert(tk.END, F"        Summary {selected_model}\n")
        summary_widget.insert(tk.END, summary_text)

        # Text box'un yüksekliğini ayarlayarak fazla boşlukları kaldırın
        summary_widget.config(height=len(summary_text.split("\n"))+3)  # Örnek olarak, maksimum 10 satır

        summary_widget.pack(padx=75, pady=20, expand=True, fill='x')



    def translate_creator():
        # Function to trigger translation
        selected_language = language_combobox.get()
        language_codes = {
            'English': 'en',
            'Turkish': 'tr',
            'French': 'fr',
            'German': 'de',
            'Spanish': 'es',
        }
        
        translator = Translator()
        dest_language_code = language_codes.get(selected_language)
        time.sleep(1)
        translated = translator.translate(combined_content, src='en', dest=dest_language_code)
        
        # Remove trailing whitespace and empty lines from translated text
        translated_text = "\n".join(line.strip() for line in translated.text.split("\n") if line.strip())

        translate_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
        translate_widget.insert(tk.END, f"        Translated Sentence : {selected_language}\n")
        translate_widget.insert(tk.END, translated_text)
        translate_widget.pack(padx=75, pady=20, expand=True, fill='x')


    # Adding the button
    summary_button = tk.Button(frame, text="Summary", command=summary_creator)
    summary_button.pack(pady=10)

    # Add the Combobox to select language
    models = ['xsum', 'pegasus-daily-maily', "bart-daily-maily"]  # models
    selected_model = tk.StringVar()
    model_combobox = ttk.Combobox(frame, values=models, textvariable=selected_model)
    model_combobox.pack(pady=10)

    # Add the Combobox to select language
    languages = ['English', 'French', 'German', 'Spanish', "Turkish"]  # Example languages
    selected_language = tk.StringVar()
    language_combobox = ttk.Combobox(frame, values=languages, textvariable=selected_language)
    language_combobox.pack(pady=10)

    translate_button = tk.Button(frame, text="Translate", command=translate_creator)
    translate_button.pack(pady=10)

    # Add Scrollbars to Canvas
    scrollbar_y = tk.Scrollbar(text_window, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar_y.set)

    scrollbar_x = tk.Scrollbar(text_window, orient=tk.HORIZONTAL, command=canvas.xview)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.configure(xscrollcommand=scrollbar_x.set)

    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox('all'))

    frame.bind('<Configure>', on_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    text_window.bind('<MouseWheel>', _on_mousewheel)

def extract_text_from_image(file_path):
    # Function to extract text from an image
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # dilation parameter , bigger means less rect
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Creating a copy of image
    im2 = gray.copy()

    cnt_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(im2, (x, y), 8, (255, 255, 0), 8)

        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]
       
        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)

        cnt_list.append([x, y, text])
        
    # A text file is created
    file = open("recognized.txt", "w+")
    file.write("")
    file.close()

    # Sort the list with respect to their coordinates, in order from top to bottom
    sorted_list = sorted(cnt_list, key=lambda x: x[1])

    # Open the file in write mode to clear previous content
    with open("recognized.txt", "w") as file:
        # Write sorted text into the file
        for x, y, text in sorted_list:
            file.write(text.strip() + "\n")

    # Initialize an empty string to store the combined content
    combined_content = ""

    # Open the file in read mode
    with open("recognized.txt", "r") as file:
        # Read lines one by one and concatenate them
        for line in file:
            combined_content += line.strip() + "\n"

    return combined_content

root = tk.Tk()
root.title("OpenCV Image Viewer")
root.geometry(f"{MAX_WINDOW_WIDTH}x{MAX_WINDOW_HEIGHT}")  # Fixed window size

image_button = tk.Button(root, text="Open Image", command=open_file)
image_button.grid(row=0, column=0, pady=10)


text_button = tk.Button(root, text="Extract Text", command=extract_text)

root.mainloop()
