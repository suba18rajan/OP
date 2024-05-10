import tensorflow as tf
import numpy as np
from tkinter import filedialog
from tkinter import *
import os
from tkinter import filedialog
import cv2
import time
from matplotlib import pyplot as plt
from tkinter import messagebox

def endprogram():
	print ("\nProgram terminated!")
	sys.exit()
def training1():
    import train as tr

def training():



    global training_screen
    training_screen = Toplevel(main_screen)
    training_screen.title("Training")
    # login_screen.geometry("400x300")
    training_screen.geometry("600x450+650+150")
    training_screen.minsize(120, 1)
    training_screen.maxsize(1604, 881)
    training_screen.resizable(1, 1)
    # login_screen.title("New Toplevel")

    Label(training_screen, text='Upload Image ', background="#d9d9d9", disabledforeground="#a3a3a3",
          foreground="#000000", bg="turquoise", width="300", height="2", font=("Calibri", 16)).pack()
    Label(training_screen, text="").pack()
    Label(training_screen, text="").pack()
    Label(training_screen, text="").pack()
    Button(training_screen, text='Upload Image', font=(
        'Verdana', 15), height="2", width="30", command=imgtraining).pack()

def imgtraining():
    import_file_path = filedialog.askopenfilename()

    image = cv2.imread(import_file_path)
    filename = 'Test.jpg'
    cv2.imwrite(filename, image)
    print("After saving image:")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original image', image)
    cv2.imshow('Gray image', gray)
    # import_file_path = filedialog.askopenfilename()
    print(import_file_path)
    fnm = os.path.basename(import_file_path)
    print(os.path.basename(import_file_path))

    from PIL import Image, ImageOps

    im = Image.open(import_file_path)
    im_invert = ImageOps.invert(im)
    im_invert.save('lena_invert.jpg', quality=95)
    im = Image.open(import_file_path).convert('RGB')
    im_invert = ImageOps.invert(im)
    im_invert.save('tt.png')
    image2 = cv2.imread('tt.png')
    cv2.imshow("Invert", image2)

    """"-----------------------------------------------"""

    img = image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original image', img)
    #cv2.imshow('Gray image', gray)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv2.imshow("Nosie Removal", dst)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    print("\n*********************\nImage : " + fnm + "\n*********************")
    img = cv2.imread(import_file_path)
    if img is None:
        print('no data')

    img1 = cv2.imread(import_file_path)
    print(img.shape)
    img = cv2.resize(img, ((int)(img.shape[1] / 5), (int)(img.shape[0] / 5)))
    original = img.copy()
    neworiginal = img.copy()
    cv2.imshow('original', img1)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original image', img1)
   # cv2.imshow('Gray image', gray)
    p = 0
    for i in range(img.shape[0]):

        for j in range(img.shape[1]):
            B = img[i][j][0]
            G = img[i][j][1]
            R = img[i][j][2]
            if (B > 110 and G > 110 and R > 110):
                p += 1

    totalpixels = img.shape[0] * img.shape[1]
    per_white = 100 * p / totalpixels
    if per_white > 10:
        img[i][j] = [500, 300, 200]
        cv2.imshow('color change', img)
    # Guassian blur
    blur1 = cv2.GaussianBlur(img, (3, 3), 1)
    # mean-shift algo
    newimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    img = cv2.pyrMeanShiftFiltering(blur1, 20, 30, newimg, 0, criteria)
    cv2.imshow('means shift image', img)
    # Guassian blur
    blur = cv2.GaussianBlur(img, (11, 11), 1)
    cv2.imshow('Noise Remove', blur)
    corners = cv2.goodFeaturesToTrack(gray, 27, 0.01, 10)
    corners = np.int0(corners)

    # we iterate through each corner,
    # making a circle at each point that we think is a corner.
    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 3, 255, -1)

    plt.imshow(image), plt.show()


def getimage1():
    import_file_path = filedialog.askopenfilename()

    image = cv2.imread(import_file_path)
    cv2.imshow('Original image', image)
    image = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Resized', image)
    cv2.imshow('Gray image', gray)
    # import_file_path = filedialog.askopenfilename()
    print(import_file_path)
    fnm = os.path.basename(import_file_path)
    print(os.path.basename(import_file_path))
    cv2.imwrite(fnm, image)

    from PIL import Image, ImageOps

    im = Image.open(import_file_path)
    im_invert = ImageOps.invert(im)
    im = Image.open(import_file_path).convert('RGB')
    im_invert = ImageOps.invert(im)

    """"-----------------------------------------------"""

    img = image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original image', img)
    cv2.imshow('Gray image', gray)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv2.imshow("Nosie Removal", dst)
    import numpy as np
    from skimage import feature, io
    from sklearn import preprocessing

    img = io.imread(fnm, as_gray=True)

    S = preprocessing.MinMaxScaler((0, 11)).fit_transform(img).astype(int)
    Grauwertmatrix = feature.greycomatrix(S, [1, 2, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=12,
                                          symmetric=False, normed=True)

    ContrastStats = feature.greycoprops(Grauwertmatrix, 'contrast')
    CorrelationtStats = feature.greycoprops(Grauwertmatrix, 'correlation')
    HomogeneityStats = feature.greycoprops(Grauwertmatrix, 'homogeneity')
    # print(ContrastStats)
    ASMStats = feature.greycoprops(Grauwertmatrix, 'ASM')

    glcm = [np.mean(ContrastStats), np.mean(CorrelationtStats), np.mean(ASMStats), np.mean(HomogeneityStats)]
    print("Feature Point:" + str(glcm))
    import numpy as np  # linear algebra
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
    # import kerastuner as kt
    # from tensorflow import keras
    # import tensorflow as tf
    # from kerastuner.tuners import RandomSearch
    # from kerastuner.engine.hyperparameters import HyperParameter as hp
    from keras.layers import Dense, Dropout, Activation, Add, MaxPooling2D, Conv2D, Flatten
    from keras.models import Sequential
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import load_model
    import numpy as np
    # import matplotlib.pyplot as plt
    # from keras.applications import VGG19
    # from keras import layers
    from keras.preprocessing import image
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    class_names = {0: "Fake", 1: "Real"}
    # example 1
    MODEL_PATH = 'model.h5'
    # Load your trained model
    model = load_model(MODEL_PATH)
    image_path = fnm
    new_img = image.load_img(image_path, target_size=(244, 244))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    prediction = np.argmax(prediction, axis=1)
    print(prediction)
    print(class_names[prediction[0]])
    ss = class_names[prediction[0]]
    messagebox.showinfo("Result", ss)

def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 600
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    # main_screen.geometry("300x250")
    main_screen.title("HANDWRITTEN SIGNATURE VERIFICATION")

    Label(text="HANDWRITTEN SIGNATURE VERIFICATION", bg="turquoise", width="300", height="5", font=("Calibri", 16)).pack()

    Button(text="Training", font=(
        'Verdana', 15), height="2", width="30", command=training, highlightcolor="black").pack(side=TOP)
    Label(text="").pack()
    Button(text="FullTraining", font=(
        'Verdana', 15), height="2", width="30", command=training1, highlightcolor="black").pack(side=TOP)
    Label(text="").pack()
    Button(text="Testing", font=(
        'Verdana', 15), height="2", width="30", command=getimage1).pack(side=TOP)

    Label(text="").pack()

    main_screen.mainloop()
main_account_screen()

