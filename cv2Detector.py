import cv2
import numpy as np
import keras

model = keras.saving.load_model("model.keras")

feed = cv2.VideoCapture(0)


txtLI = ["smoking","not smoking"] 
def txt(pos,raw):
    cv2.putText(frame, f"{txtLI[pos]}: {raw}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

while True:
    vidBool , frame = feed.read()

    if not vidBool:
        break

    img = cv2.resize(frame,(244,244))
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_norm = img_rgb/255
    tensorInput = np.expand_dims(img_norm,0)

    prediction = model.predict(tensorInput)

    rounded = int(prediction[0] > 0.5)

    txt(rounded,prediction[0])

    cv2.imshow("detector",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

feed.release()
cv2.destroyAllWindows()
