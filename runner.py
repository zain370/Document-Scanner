import tkinter as tk
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from imutils.perspective import four_point_transform
from pyimagesearch.transform import four_point_transform
from PIL import Image, ImageTk
import img2pdf

# Imported Libraries for operations


past = 196
# Initializing Tkinter Application
app = Tk()
app.title('Document Scanner')
app.geometry('650x750')

# Adding 'File' menu bar to the menu section
menubar = tk.Menu()
file_menu = tk.Menu(menubar, tearoff=False)
edit_menu = tk.Menu(menubar, tearoff=False)

# File menu bar items added
file_menu.add_command(label="Browse", command=lambda: selectImage())
file_menu.add_command(label="Capture", command=lambda: capture_image())
file_menu.add_command(label="Exit", command=lambda: app.quit())

menubar.add_cascade(menu=file_menu, label="File")
app.config(menu=menubar)

# Binding escape key to the application to quit
app.bind('<Escape>', lambda e: app.quit())
label_widget = Label(app)
width, height = 800, 600
WIDTH, HEIGHT = 1920, 1080

# Initializing video capture from default device camera with set height and width
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
file_path = ''


def capture_image():
    # Enables/Disables buttons based upon image capture from camera
    file_menu.entryconfig(1, state=DISABLED)
    file_menu.entryconfig(2, state=DISABLED)

    # Captures frame from the camera
    result, frame = vid.read()

    if result:
        cv2.imwrite("input.jpg", frame)
        image = Image.open('input.jpg')
        image = image.resize((250, 250), Image.BILINEAR)
        test = ImageTk.PhotoImage(image)
        label1 = tk.Label(image=test)
        label1.image = test
        label1.size = image.size
        label1.place(x=0, y=200)

        def deleteImage():
            label1.config(image='')
            delete_button.destroy()
            file_menu.entryconfig(1, state=NORMAL)
            file_menu.entryconfig(2, state=NORMAL)

        delete_button = tk.Button(app, text="Delete", command=deleteImage)
        delete_button.place(x=100, y=220)


def selectImage():
    # Open image browser dialog
    filename = filedialog.askopenfilename()

    print(filename)
    if filename:
        file_menu.entryconfig(1, state=DISABLED)
        file_menu.entryconfig(2, state=DISABLED)
        image = Image.open(filename)
        img = cv2.imread(filename)
        cv2.imwrite("input.jpg", img)
        image = image.resize((250, 250), Image.BILINEAR)
        test = ImageTk.PhotoImage(image)
        label1 = tk.Label(image=test)
        label1.image = test
        label1.size = image.size
        label1.place(x=0, y=200)

        def deleteImage():
            label1.config(image='')
            delete_button.destroy()
            file_menu.entryconfig(1, state=NORMAL)
            file_menu.entryconfig(2, state=NORMAL)

        delete_button = tk.Button(app, text="Delete", command=deleteImage)
        delete_button.place(x=100, y=220)


def blur_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    result = cv2.GaussianBlur(gray, (blur_scale.get(), blur_scale.get()), 0)
    return result


def threshHold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    thresh = cv2.threshold(gray, thresh_high_scale.get(),
                           thresh_low_scale.get(), cv2.THRESH_BINARY)
    result = cv2.cvtColor(thresh[1], cv2.COLOR_BGRA2GRAY)
    return result


def scan_detection(image):
    global document_contour
    document_contour = np.array(
        [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    # color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # highlight white regions and convert to grey scale
    result = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(
        gray, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # find largest by sorting
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area
    cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 3)
    warped = four_point_transform(image, document_contour.reshape(4, 2))
    return cv2.resize(warped, (int(warped.shape[1]), int(warped.shape[0])))


def denoise_image(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(
        image, None, 10, 10, 7, 21)
    contrast_stretched_image = cv2.normalize(
        denoised_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    sharpened_image = cv2.filter2D(contrast_stretched_image, -1, kernel=kernel)
    brightness_image = cv2.convertScaleAbs(sharpened_image, alpha=1, beta=5)

    lookup_table = np.array([((i / 255.0) ** gamma_scale.get())
                            * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected_image = cv2.LUT(brightness_image, lookup_table)
    return gamma_corrected_image


def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def save_pdf():
    image = Image.open('output.jpg')
    pdf_bytes = img2pdf.convert(image.filename)
    file = open('output.pdf', "wb")
    file.write(pdf_bytes)
    image.close()
    file.close()


def brightness(image):
    return cv2.convertScaleAbs(image, image, brightness_scale.get())


def fixBlur(n):
    # Sets blur scale value to odd for filter. The user cannot enter even values for filter
    # 3x3, 5x5, 7x7 and not 2x2, 4x4, 6x6
    global past
    n = int(n)
    if not n % 2:
        blur_scale.set(n + 1 if n > past else n - 1)
        past = blur_scale.get()


# Variable to carry operations
blurBool = IntVar()
threshBool = IntVar()
scanBool = IntVar()
denoiseBool = IntVar()
inverseBool = IntVar()
sharpenBool = IntVar()
brightBool = IntVar()


def applyFilters():
    # This func is called upon 'Apply Filter'
    # Filters are applied sequentially. However the sequence can be changed by re-ordering the functions
    image = cv2.imread('input.jpg')

    # Checks if the image exists or not
    if image is None:
        return
    if scanBool.get() == 1:
        # Scans the documents and separates from the background
        image = scan_detection(image)

    if denoiseBool.get() == 1:
        # Denoise filter
        image = denoise_image(image)

    if threshBool.get() == 1:
        # Threshold filter with two sliders for upper and lower bound
        image = threshHold(image)

    if blurBool.get() == 1:
        # Blur filter
        image = blur_image(image)

    if inverseBool.get() == 1:
        # Image inverse filter
        image = 255 - image

    if sharpenBool.get() == 1:
        # Image sharpening filter
        image = sharpen_image(image)

    if brightBool.get() == 1:
        # Brightness Filter
        image = brightness(image)

    # Writes output image to the default directory and shows image on screen
    cv2.imwrite('output.jpg', image)
    cv2.imshow("Output Image",
               cv2.resize(image, (int(size_scale.get() * image.shape[1]), int(size_scale.get() * image.shape[0]))))


# For packing check buttons and sliders into the Tkinter Application
Label(textvariable=tk.StringVar(value="Blur")).grid(row=0, column=0)
Checkbutton(app, text="Blur", variable=blurBool,
            onvalue=1, offvalue=0).grid(row=1, column=0)
blur_scale = tk.Scale(from_=1, to_=101, command=fixBlur, orient=tk.HORIZONTAL)
blur_scale.grid(row=2, column=0)

Label(textvariable=tk.StringVar(value="Threshold")).grid(row=0, column=1)
Checkbutton(app, text="Thresh", variable=threshBool,
            onvalue=1, offvalue=0).grid(row=1, column=1)
thresh_low_scale = tk.Scale(from_=1, to_=255, orient=tk.HORIZONTAL)
thresh_low_scale.set(255)
thresh_low_scale.grid(row=2, column=1)
thresh_high_scale = tk.Scale(from_=1, to_=255, orient=tk.HORIZONTAL)
thresh_high_scale.set(155)
thresh_high_scale.grid(row=3, column=1)

Label(textvariable=tk.StringVar(value="Image Scale")).grid(row=0, column=2)
size_scale = tk.Scale(from_=0.1, to_=1, resolution=0.1, orient=tk.HORIZONTAL)
size_scale.set(0.5)
size_scale.grid(row=1, column=2)

Label(textvariable=tk.StringVar(value="Gamma")).grid(row=0, column=3)
Checkbutton(app, text="Denoise", variable=denoiseBool,
            onvalue=1, offvalue=0).grid(row=1, column=3)
gamma_scale = tk.Scale(from_=0.1, to_=5, resolution=0.1, orient=tk.HORIZONTAL)
gamma_scale.set(1.5)
gamma_scale.grid(row=2, column=3)

Label(textvariable=tk.StringVar(value="Brightness")).grid(row=0, column=4)
Checkbutton(app, text="Brightness", variable=brightBool,
            onvalue=1, offvalue=0).grid(row=1, column=4)
brightness_scale = tk.Scale(from_=1, to_=255, orient=tk.HORIZONTAL)
brightness_scale.set(1)
brightness_scale.grid(row=2, column=4)

Checkbutton(app, text="Scan", variable=scanBool,
            onvalue=1, offvalue=0).grid(row=0, column=5)
Checkbutton(app, text="Sharpen", variable=sharpenBool,
            onvalue=1, offvalue=0).grid(row=1, column=5)
Checkbutton(app, text="Inverse", variable=inverseBool,
            onvalue=1, offvalue=0).grid(row=3, column=5)

Button(app, text="Apply Filters", command=applyFilters).grid(row=4, column=0)
Button(app, text="Save Pdf", command=save_pdf).grid(row=4, column=1)

app.mainloop()
