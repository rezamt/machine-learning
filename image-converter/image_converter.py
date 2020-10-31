# https://pillow.readthedocs.io/en/stable/handbook/index.html
# The Python Imaging Library adds image processing capabilities to your Python interpreter.
from PIL import Image


def process_image(image_name):
    img = Image.open('eggs.png').convert('L')  # convert image to 8-bit grayscale

    WIDTH, HEIGHT = img.size

    data = list(img.getdata())  # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

    # At this point the image's pixels are all in memory and can be accessed
    # individually using data[row][col].

    # For example:
    for row in data:
        print(' '.join('{:3}'.format(value) for value in row))

    # Here's another more compact representation.
    chars = '@%#*+=-:. '  # Change as desired.
    scale = (len(chars) - 1) / 255.
    print()
    for row in data:
        print(' '.join(chars[int(value * scale)] for value in row))


if __name__ == '__main__':
    image_name = 'eggs.png'
    process_image(image_name)
