import copy
import cv2
from matplotlib import pyplot as plt

# DATASET CREATION TO USE IT LATER IN KNN & CNN

videoPath = './datasets/customDataset/video/'
imgPath = './datasets/customDataset/img TEMPORAL/'
facePath = './datasets/customDataset/face/'
eyesPath = './datasets/customDataset/eyes/'
def formVideoToImages(videoName):
    # Obtenim imgs del video
    video = cv2.VideoCapture(videoPath + videoName + '.MOV')

    currentFrame = 1
    count = 1
    while(True):
        ret, frame = video.read()
        if ret:
            if currentFrame % 8 == 0:
                imgName = imgPath + videoName + '_' + str(count) + '.jpg'
                cv2.imwrite(imgName, frame)
                count += 1
                '''print("Image", imgName, " created")'''
            currentFrame += 1
        else:
            break

    print("Count: ", count)
    video.release()
    cv2.destroyAllWindows()

    return count

def detectEyesFromCamera():
    face_cascade = cv2.CascadeClassifier('./preTrainedModels/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./preTrainedModels/haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectEyesFromImg(img, imgName, parcialPath, face_cascade, eye_cascade):
    img_face = copy.deepcopy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    count_faces = 1
    count_eyes = 1
    print(faces)
    (x, y, w, h) = faces[0]
    cv2.rectangle(img_face, (x, y), (x + w, y + h), (255, 0, 0), 2)
    f = img_face[y:y + h, x:x + w]
    print(parcialPath + '.jpg')
    cv2.imwrite(parcialPath + '.jpg', f)
    plt.imshow(f)
    plt.show()
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        print('hello')
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        e = f[ey:ey + eh, ex:ex + ew]
        #print(eyesPath + parcialPath + '_' + str(count_eyes) + '.jpg')
        cv2.imwrite(parcialPath + '.jpg', e)
        plt.imshow(e)
        plt.show()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    imgName = 'dina' # 'dina'
    #categories = {'center': 26, 'up': 11, 'down': 12, 'left': 18, 'right': 13, 'close': 3} #falta img_right_10
    face_cascade = cv2.CascadeClassifier('./preTrainedModels/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./preTrainedModels/haarcascade_eye.xml')
    categories = {'left': 28}
    # Convert Video recorded into images
    #imgCount = formVideoToImages(imgName)
    '''img = cv2.imread(imgPath + './datasets/customDataset/' + '.jpg')
    detectEyesFromImg(img, imgName, img, face_cascade, eye_cascade)'''

    # Iterate in the images created
    '''for key, value in categories.items():
        for i in range(15, value):
            parcialPath = key + '/' + key + '_' + str(i)
            print(str(key), str(i))
            img = cv2.imread(imgPath + parcialPath + '.jpg')
            print(imgPath)
            print(parcialPath)
            detectEyesFromImg(img, key + '_' + str(i), parcialPath, face_cascade, eye_cascade)
    '''
    # Detect eyes
    img = cv2.imread(imgPath + "felix.jpg")
    plt.imshow(img)
    plt.show()
    detectEyesFromImg(img, "center", './datasets/customDataset/test/img TEMPORAL', face_cascade, eye_cascade)

    plt.show()
