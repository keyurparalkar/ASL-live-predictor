from flask import Flask
import fastai
from fastai import *
from fastai.vision import *
import pathlib
# import cv2

app = Flask(__name__)

path = pathlib.PosixPath('./data/asl-alphabet/')
data = ImageDataBunch.from_folder(path=path,ds_tfms=get_transforms(),size=224,valid_pct=0.3)
data.normalize(imagenet_stats)
learn = create_cnn(data,models.resnet34,metrics=accuracy)

learn.load('stage-1')

@app.route("/")
def hello_world():
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()   #capture each frame
    #     ret = cap.set(3,224)
    #     ret = cap.set(4,224)

        # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   #converts each frame to gray

        cv2.imshow('FRAME',frame)
        # cv2.imshow('GRAY',gray)

        #For capturing frame and saving it as an image at given folder:
        if cv2.waitKey(1) == ord('n'):
            cv2.imwrite('test1.jpg',frame)
            img = open_image(pathlib.PosixPath('./test1.jpg'))
            print(learn.predict(img))
        
        #making actual prediction for each frame:=============================================
        #For quitting the given session: ord is used to obtain unicode of the given string.
        #cv2.waitKey returns the unicode of the key which is pressed
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

