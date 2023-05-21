import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img = plt.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
    return images

if __name__ == '__main__':
    datasetPath = './datasets/customDataset/eyes/'
    positionList = {'center', 'close', 'down', 'left', 'right', 'up'}
    imgDict = {}
    X = []
    y = []

    for i in positionList:
        imgDict[i] = load_images_from_folder(datasetPath+i)

    #img_test = cv2.cvtColor(imgDict['center'][0], cv2.THRESH_BINARY)
    #img_test = resize(img_test, (128*4, 64*4))
    #img_test = resize(imgDict['center'][0], (128*4, 64*4))


    fd, hog_image = hog(imgDict['center'][0], orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, channel_axis=-1)

    """
    plt.figure()
    plt.imshow(hog_image, cmap='gray')
    
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,10))
    plt.figure()
    plt.imshow(hog_image_rescaled, cmap='gray')"""

    for i in positionList:
        for j in range(len(imgDict[i])):
            resized_image = resize(imgDict[i][j], (28*4,28*4))
            fd, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True, channel_axis=-1)
            X.append(exposure.rescale_intensity(hog_image, in_range=(0,10)))
            y.append(i)

    X = np.array(X)
    X = np.reshape(X,(X.shape[0], (len(X[1])*len(X[2]))))
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.25,
                                                        random_state = 1,
                                                        stratify = y)

    sc = StandardScaler()
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    ## RANDOM FOREST CLASSIFIER
    modelRF = RandomForestClassifier()
    modelRF.fit(X_train, y_train)

    print('RANDOM FOREST CLASSIFIER')
    print('Train Accuracy: '+ str(round(modelRF.score(X_train, y_train), 2)))
    print('Test Accuracy: '+ str(round(modelRF.score(X_test, y_test), 2)))

    y_predRF = modelRF.predict(X_test_std)
    accuracy_score(y_predRF, y_test)
    print(classification_report(y_predRF, y_test))

    ## ONE VS REST REGRESSION MULTI CLASS CLASSIFICATION

    modelOVA = LogisticRegression(multi_class='ovr')
    modelOVA.fit(X_train, y_train)

    print('ONE VS REST REGRESSION')
    print('Train Accuracy: ' + str(round(modelOVA.score(X_train, y_train), 2)))
    print('Test Accuracy: ' + str(round(modelOVA.score(X_test, y_test), 2)))

    y_predOVA = modelOVA.predict(X_test_std)
    accuracy_score(y_predOVA, y_test)
    print(classification_report(y_predOVA, y_test))

    ## SUPPORT VECTOR MACHINE

    modelSVM = SVC(kernel='poly', degree=1)
    modelSVM.fit(X_train, y_train)

    print('SUPPORT VECTOR MACHINE')
    print('Train Accuracy: ' + str(round(modelSVM.score(X_train, y_train), 2)))
    print('Test Accuracy: ' + str(round(modelSVM.score(X_test, y_test), 2)))

    y_predSVM = modelSVM.predict(X_test_std)
    accuracy_score(y_predSVM, y_test)
    print(classification_report(y_predSVM, y_test))





    

