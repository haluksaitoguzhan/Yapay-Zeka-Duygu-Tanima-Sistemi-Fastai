import cv2
from fastai.vision.all import *
import os

def pred(img):
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model.pkl')
    learn = load_learner(model_path)

    prediction, predClass, probabilities = learn.predict(img)

    return prediction, predClass, probabilities

def kafayiSec(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    return faces

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and int(max_width) - 20 <= x <= int(max_width) and 0 <= y <= 20:
        # Sağ üst köşedeki X simgesine tıklandığında kamerayı kapat
        cap.release()
        cv2.destroyAllWindows()

# Kamerayı açın ve kameranın maksimum çözünürlüğünü alın
cap = cv2.VideoCapture(0)
max_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
max_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Fare tıklamalarını takip et
cv2.namedWindow("Kamera")
cv2.setMouseCallback("Kamera", on_mouse_click)

while True:
    ret, frame = cap.read()
    image = cv2.resize(frame, (int(max_width), int(max_height)))
    faces = kafayiSec(image)

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_image = image[y:y + h, x:x + w]
        prediction, predClass, probabilities = pred(face_image)
        cv2.putText(image, f"Tahmin: {prediction}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


    # Sağ üst köşeye çarpı işareti koy
    cv2.putText(image, 'X', (int(max_width) - 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Görüntüyü ekranda göster
    cv2.imshow("Kamera", image)

    # Kullanıcıdan bir tuş basmasını bekleyin.
    k = cv2.waitKey(1)

    # 'q' tuşuna basılırsa döngüyü sonlandırın.
    if k == ord('q'):
        break

# Kamerayı kapatın.
cap.release()
cv2.destroyAllWindows()