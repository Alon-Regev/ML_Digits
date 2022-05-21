from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from NeuralNetwork import NeuralNetwork

def main():
    # open image as grayscale
    img = Image.open("pic.jpeg").convert("L")
    # process image
    data = process_image(np.array(img.copy()))
    # load neural network
    nn = NeuralNetwork([784, 128, 10])
    nn.load("nn.npz")
    
    # get digits from picture
    digit_img = get_digits_from_picture(data)
    # draw rectangle over symbols
    draw = ImageDraw.Draw(img)
    for i in digit_img:
        symbol = i[2]
        # check symbol size
        if symbol.shape[0] < 50 or symbol.shape[1] < 50:
            continue
        # thick rectangle
        draw.rectangle((i[0], i[1], i[0] + symbol.shape[1], i[1] + symbol.shape[0]), outline="#777", width=2)
        symbol = resize_symbol(symbol)
        # predict digit
        prediction = nn.predict(symbol.reshape(784))
        # draw digit
        x, y = i[0], i[1] - 30
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        font = ImageDraw.ImageFont.truetype("arial.ttf", 24)
        draw.text((x, y), f"{digit}  {int(100 * confidence)}%", fill="#000", font=font)
    img.show()

def process_image(img):
    """
    function processes image before digit recognition
    input: 
        img: numpy array of image
    return: 
        numpy array of image
    """
    # turn bright pixels to white
    img[img > 125] = 255
    # blur and darken image
    img = Image.fromarray(img)
    img = img.filter(ImageFilter.GaussianBlur(radius=5))
    img = img.point(lambda p: max(0, 255 - 2.5 * (255 - p)))
    return np.array(img)
    
def get_digits_from_picture(img):
    """
    function finds a digit from a picture
    input: 
        img: numpy array of gray scale image
    return: 
        digit_img: numpy array of digit
    """
    result = []
    # go over picture top-down left-right and find non-white pixel
    y = 0
    while y < img.shape[0]:
        x = 0
        while x < img.shape[1]:
            if img[y, x] != 255:
                # get digit image
                rx, ry, w, h, symbol = extract_symbol(img, x, y)
                result.append((rx, ry, symbol))
                # delete pixels
                img[ry:ry+h, rx:rx+w] = 255
            x += 1
        y += 1
    return result

def extract_symbol(img, x, y, w=4, h=4):
    """
    function seperates the symbol from a picture
    input: 
        img: numpy array of gray scale image
        x, y: position where the symbol was found
    return: 
        symbol: numpy array of symbol
    """
    # get border
    top = img[y, x:x+w]
    bottom = img[y+h, x:x+w]
    left = img[y:y+h, x]
    right = img[y:y+h, x+w]
    # check if border intersects white pixels
    try_again = False
    if any(top != 255):
        y -= 1
        h += 1
        try_again = True
    if any(bottom != 255):
        h += 1
        try_again = True
    if any(left != 255):
        x -= 1
        w += 1
        try_again = True
    if any(right != 255):
        w += 1
        try_again = True
    # if border intersects, try again
    if try_again:
        return extract_symbol(img, x, y, w, h)
    else:
        return x, y, w, h, img[y:y+h+1, x:x+w+1].copy()

def resize_symbol(img):
    """
    function resizes image to correct size for digit recognition
    input: 
        img: numpy array of symbol (any size, grayscale)
    return: 
        numpy array of symbol (28x28 pixels)
    """
    # resize image to 28x28
    img = Image.fromarray(img)
    img.thumbnail((24, 24), Image.ANTIALIAS)
    # paste on white background
    back = Image.new("L", (28, 28), 255)
    back.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
    # invert and return as data
    return 1 - np.array(back) / 255

if __name__ == "__main__":
    main()