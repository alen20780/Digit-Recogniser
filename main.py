import cv2
import numpy as np
import nn_model
import svm_model
import knn_model

startInference = False
def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

threshold = 100
def on_threshold(x):
    global threshold
    threshold = x

def start_cv(model, scaler, predict_func, is_nn=False):
    global threshold
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)
    cv2.createTrackbar('threshold', 'background', 150, 255, on_threshold)
    background = np.zeros((480, 640), np.uint8)
    frameCount = 0

    while True:
        ret, frame = cap.read()

        if startInference:
            frameCount += 1
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)
            resizedFrame = thr[240-75:240+75, 320-75:320+75]
            background[240-75:240+75, 320-75:320+75] = resizedFrame

            iconImg = cv2.resize(resizedFrame, (28, 28))
            if is_nn:
                res = predict_func(model, iconImg)
            else:
                res = predict_func(model, scaler, iconImg)

            if frameCount == 5:
                background[0:480, 0:80] = 0
                frameCount = 0

            cv2.putText(background, res, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.rectangle(background, (320-80, 240-80), (320+80, 240+80), (255, 255, 255), thickness=3)
            cv2.imshow('background', background)
        else:
            cv2.imshow('background', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Choose the model to train and use:")
    print("1. Neural Network")
    print("2. Support Vector Machine")
    print("3. K-Nearest Neighbors")

    valid_choice = False
    while not valid_choice:
        try:
            choice = int(input("Enter the number of your choice: "))
            if choice in [1, 2, 3]:
                valid_choice = True
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number (1, 2, or 3).")

    if choice == 1:
        (x_train, y_train, x_test, y_test) = nn_model.get_mnist_data()
        try:
            model = nn_model.load_model('nn_model.keras')
            print('Loaded saved neural network model.')
        except:
            print("Training neural network model...")
            model = nn_model.train_model(x_train, y_train, x_test, y_test)
            nn_model.save_model(model, 'nn_model.keras')
        start_cv(model, None, nn_model.predict, is_nn=True)

    elif choice == 2:
        (x_train, y_train, x_test, y_test) = svm_model.get_mnist_data()
        print("Training SVM model...")
        model, scaler = svm_model.train_model(x_train, y_train)
        start_cv(model, scaler, svm_model.predict)

    elif choice == 3:
        (x_train, y_train, x_test, y_test) = knn_model.get_mnist_data()
        print("Training KNN model...")
        model, scaler = knn_model.train_model(x_train, y_train)
        start_cv(model, scaler, knn_model.predict)

    else:
        print("Invalid choice. Exiting.")

if __name__ == '__main__':
    main()