from PIL import Image, ImageDraw
import numpy as np

def main():
    # open image as grayscale
    img = Image.open("pic.png").convert("L")
    # to numpy
    #img = np.array(img)
    
    digit_img = get_digits_from_picture(np.array(img.copy()))
    # draw rectangle over symbols
    draw = ImageDraw.Draw(img)
    for i in digit_img:
        draw.rectangle((i[0], i[1], i[0] + i[2].shape[1], i[1] + i[2].shape[0]), outline="#999")
    
    img.show()

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

if __name__ == "__main__":
    main()