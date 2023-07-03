from faceDetection import FaceDetection
from faceRecognition import FaceRecognition

# Detects and extract each face
app = FaceDetection()
app.extract_face()


# Face recognition using knn
app1 = FaceRecognition()
app1.test()
