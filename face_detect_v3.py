import cv2
import os
import json


def detectface(imagePath):
    # CONST_SCALEFACTOR Parameter specifying how much the image size is
    # reduced at each image scale.
    #
    # The bigger the number: faster, less precise
    #
    # Should ALWAYS be greater than 1 (x > 1)
    #
    # You may increase it to as much as 1.4 for faster detection,
    # Recommended value is 1.05
    CONST_SCALEFACTOR = 1.2

    # CONST_MINNEIGHBORS Parameter specifying how many neighbors each
    # candidate rectangle should have to retain it.
    # This parameter will affect the quality of the detected faces
    # Higher value results in less detections but with higher quality
    # Recommended value is 3 - 6
    CONST_MINNEIGHBORS = 3
    # CONST_MINSIZE Parameter specifying minimum possible object size.
    # Objects smaller than that are ignored.
    # Recommended value is [100, 100]
    CONST_MINSIZE = (100, 100)

    OUTPUT_FOLDER = os.path.join(os.getcwd(), "output\\")
    cascPath = "haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)

    image = cv2.imread(imagePath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=CONST_SCALEFACTOR,
        minNeighbors=CONST_MINNEIGHBORS,
        minSize=CONST_MINSIZE
    )
    output_json = {
        'image_path': imagePath,
        'folder_output': OUTPUT_FOLDER,
        'faces': []
    }
    print("Found {0} faces!".format(len(faces)))
    if len(faces) > 0:
        for index, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crop_img = image[y:y + h, x:x + w]
            name_item = "face_" + str(index) + ".jpg"
            cv2.imwrite(os.path.join(os.getcwd(), "output\\") + name_item, crop_img)
            inf_face_json = {
                'name': name_item,
                'position': {
                    'x': x,
                    'y': y
                },
                'size': {
                    'width': w,
                    'height': h
                }
            }
            output_json["faces"].append(inf_face_json)

    return json.dumps(output_json)


imagePath = "C:/test/test1.jpg"
print detectface(imagePath)
cv2.waitKey(0)
