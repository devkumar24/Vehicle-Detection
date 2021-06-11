import numpy as np
import os
import PIL
import cv2
import math

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def show_Image(path : str = "", title :  str = ""):
	"""
	View Image using OpenCv
	"""

	image = cv2.imread(path)

	cv2.imshow(title, image)
	cv2.waitKey(0)
	cv2.DestroyAllWindows()


def showMultipleImage(path: str = "",title : str = "", no_images = 25, **kwargs):
    """
    This function will also show images but it will show only 25 images in the respective path
    """

    image_files = os.listdir(path)
    idx = 1

    for image in image_files:
        path_image = os.path.join(path, image)

        img = PIL.Image.open(path_image)
        # Resize the image 
        # By default the image is 64,64,3, but no we have resize the image to 128,128,3
        img = img.resize((128,128))

        # Create a subplot to plot the images
        plt.subplot(int(math.sqrt(no_images)),int(math.sqrt(no_images)) + 1,idx)
        plt.imshow(img)

        plt.axis("off")
        plt.title(title)
        plt.tight_layout()

        idx += 1

        if idx == no_images+1:
            break


        for key, value in kwargs.items():
            if key == "saveFig" and value == "True":
                # It will save png image
                plt.savefig("images/{}.png".format(title))



def getFeatures(images, choice = None, **kwargs):
    if choice == "HOG":
        features = list()
        key = list(kwargs.keys())
        values = list(kwargs.values())
        if ("orient" in key) and ("cellPerBlock" in key) and ("pixelsPerCell" in key):
            for image in images:
                orient = kwargs['orient']
                cellPerBlock = kwargs['cellPerBlock']
                pixelsPerCell = kwargs['pixelsPerCell']
                Hog = HOG(orient=orient, cellPerBlock=cellPerBlock, pixelsPerCell=pixelsPerCell)
                r,g,b = Hog.visualizeTestImage(image, plot = False)
                features.append(np.hstack((r,g,b)))
            
            return features
        else:
            raise InputError("Either of the 3 parameters are missing!")
    else:
        if choice == "colorSpacing" or choice == "ChoiceSpacing" or choice == "colorspacing" or choice == "Colorspacing":
            pass
        else:
            if choice == "SpatialBinning":
                pass
            else:
                raise InputError("You have Entered wrong input choice!")



def makeData(vehicle_features, non_vehicle_features):
    features = np.vstack([vehicle_features, non_vehicle_features])
    labels = np.concatenate([np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))])
    
    print("Shape of X data: {}".format(features.shape))
    print("Shape of Y data: {}".format(labels.shape))
    return features, labels



def preProcessData(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
	scaler= StandardScaler()
	scaler.fit(X_train)
	X_train_scaled= scaler.transform(X_train)
	X_test_scaled= scaler.transform(X_test)

	return (X_train,y_train), (X_test, y_test)


