from NeuralNetwork import NeuralNetwork
import PIL.Image
import numpy as np
from PIL import ImageTk
from tkinter import *
from tkinter.filedialog import askopenfilename

SCALE = 10
draw = False
erase = False

nn = None
panel = None
result = None
img = None

def main():
    global nn, panel, result, img

    # load neural network
    nn = NeuralNetwork([784, 128, 10])
    nn.load("nn.npz")
    
    root = Tk()
    root.title("Digit Drawing")
    canvas = Canvas(root)
    canvas.pack()
    # non resizable
    root.resizable(False, False)
    # create a new image
    img = PIL.Image.new("L", (28, 28))
    panel = Label(canvas)
    panel.pack(side = "bottom", fill = "both", expand = "yes")

    # add result label
    # large font
    result = Label(root, text = "Prediction: None, Confidence: 0%", font = ("Helvetica", 12))
    result.pack(side = "bottom")
    display()

    # mouse events
    panel.bind("<Button-1>", lambda e: set_draw(True))
    panel.bind("<ButtonRelease-1>", lambda e: set_draw(False))
    panel.bind("<Button-3>", lambda e: set_erase(True))
    panel.bind("<ButtonRelease-3>", lambda e: set_erase(False))
    panel.bind("<Motion>", mouse_motion)

    root.mainloop()

def display():
    """
    function to display the image on the canvas (resized)
    input:
        img: image to display
        panel: label to display image on
    return: None
    """
    global img, panel

    # scale and display
    newImg = img.resize((img.width * SCALE, img.height * SCALE), PIL.Image.NEAREST)
    imgTk = ImageTk.PhotoImage(newImg)
    panel.configure(image=imgTk)
    panel.image = imgTk

# function update draw states
def set_draw(v):
    global draw
    draw = v
def set_erase(v):
    global erase
    erase = v

def mouse_motion(e):
    global draw, erase, img

    if draw or erase:
        # paint pixel
        x, y = e.x // SCALE, e.y // SCALE
        if draw:
            # add value to nearby pixels
            if 0 <= x < img.width and 0 <= y < img.height:
                img.putpixel((x, y), 255)

            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    if i >= 0 and i < img.width and j >= 0 and j < img.height:
                        # add value to nearby pixels
                        if i == x or j == y:
                            img.putpixel((i, j), img.getpixel((i, j)) + 16)
                        else:
                            img.putpixel((i, j), img.getpixel((i, j)) + 8)
        else:
            # delete pixel and nearby pixels
            for i in range(x - 2, x + 3):
                for j in range(y - 2, y + 3):
                    if 0 <= i < img.width and 0 <= j < img.height:
                        img.putpixel((i, j), 0)
        display()
        updatePrediction()

def updatePrediction():
    """
    function to update the prediction label
    input: None
    return: None
    """
    global nn, result

    # get image as 1D array
    img_arr = np.array(img).reshape(784)
    # get prediction
    prediction = nn.predict(img_arr)
    # get confidence
    digit = np.argmax(prediction)
    confidence = prediction[digit] * 100
    # update label
    result.configure(text = f"Prediction: {digit}, Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()