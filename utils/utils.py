import numpy as np
import os
import PIL
import cv2
import math

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
		path_image = os.pth.join(path, image)

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

		if idx == no_images:
			break


		for key, value in kwargs.items():
			if key == "saveFig" and value == "True":
				# It will save png image
				plt.savefig("images/{}.png".format(title))


