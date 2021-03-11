import cv2
import numpy as np
import tensorflow.keras

from PIL import Image, ImageOps
from pynput.keyboard import Key, Controller


def screenshot():
    global cam
    cv2.imwrite('screenshot.png', cam.read()[1])


def do_action(prediction, keyboard):
    if prediction == 0:
        keyboard.release(Key.down)
        print('Do Nothing')
    elif prediction == 1:
        print('Up')
        keyboard.release(Key.down)
        keyboard.press(Key.space)
        keyboard.release(Key.space)
    else:
        print('Down')
        keyboard.press(Key.down)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    cam = cv2.VideoCapture(2)

    model = tensorflow.keras.models.load_model('keras_model.h5')

    keyboard = Controller()

    while True:
        ret, img = cam.read()

        cv2.imshow('My Camera', img)

        ch = cv2.waitKey(5)
        if ch == 27:
            break

        screenshot()

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        image = Image.open('screenshot.png')

        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(image)

        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        data[0] = normalized_image_array

        prediction = np.argmax(model.predict(data)[0])
        do_action(prediction, keyboard)

    cv2.destroyAllWindows()
