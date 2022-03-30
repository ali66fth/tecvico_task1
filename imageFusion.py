

# Importing libraries
import numpy as np
import pydicom
from PIL import Image
import cv2


# Converting Dicom Images Into JPG
im1 = pydicom.dcmread(r"C:\Users\Admin\Desktop\tecvico\10_ct-20220324T071328Z-001\10_ct\1-01.dcm")
im2 = pydicom.dcmread(r"C:\Users\Admin\Desktop\tecvico\10_ct-20220324T071328Z-001\10_ct\1-02.dcm")

float_img1 = im1.pixel_array.astype(float)
float_img2 = im2.pixel_array.astype(float)

rescaled_img1 = (np.maximum(float_img1,0)/float_img1.max())*225  #float pixels
rescaled_img2 = (np.maximum(float_img2,0)/float_img2.max())*225  

int_img1 = np.uint8(rescaled_img1)  #integer pixels
int_img2 = np.uint8(rescaled_img2)

img1 = Image.fromarray(int_img1)
img2 = Image.fromarray(int_img2 )

img1.show()
img2.show()

img1.save(r"C:\Users\Admin\Desktop\img1.jpg")
img2.save(r"C:\Users\Admin\Desktop\img2.jpg")



# making Fusion image
img1 = cv2.imread(r"C:\Users\Admin\Desktop\img1.jpg")
img2 = cv2.imread(r"C:\Users\Admin\Desktop\img2.jpg")

img2 = cv2.resize(img2,(480,331))# uniform picture size 
img1 = cv2.resize(img1,(480,331))# uniform picture size 
dst = cv2.addWeighted(img1,0.5,img2,0.5,0)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()







