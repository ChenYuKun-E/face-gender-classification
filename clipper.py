import cv2
import dir


class Clipper:

    def __init__(self):
        self.recognizer = cv2.CascadeClassifier(dir.get_path("model/haarcascade_frontalface_default.xml"))

    def crop(self, img_path):
        try:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.recognizer.detectMultiScale(gray)
            if len(faces):
                x, y, w, h = faces[0]
                c_img = img[y:y + h, x:x + w]
                return cv2.resize(c_img, (28, 28), interpolation=cv2.INTER_AREA)
        except:
            pass

        return None

    def crop_and_save(self, img_path, save_path):
        img = self.crop(img_path)
        if img is not None:
            cv2.imwrite(save_path, img)
