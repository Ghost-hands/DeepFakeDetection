Abstract:
Detecting deepfaked faces in images and videos is the primary objective of this project.
A desktop application that uses an image classification model to predict whether the
image or video chosen by the user is deepfake or not is the culmination of the work to be
done in this project. Moreover, a dataset which is a variation of the
OpenForensics dataset will be used to train, test, and validate the following
three main models: a custom CNN sequential model, a model that uses the ResNet50 pretrained model as its base, and a model that uses the MobileNet pre-trained model as its
base. Only the
model with the highest test and validation accuracies, with the general goal being to
achieve 90% accuracy in both, will be used in the backend of the desktop application.
............................................................................................................................
Link for dataset:https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
............................................................................................................................
contains all the implementation for the three main models trained, tested and validated on the dataset + the code to run the Deepfake detection GUI application + some saved .h5 models.
.............................................................................................................................
FUTURE PLANS:
-Use flask to create a web application of this work.
