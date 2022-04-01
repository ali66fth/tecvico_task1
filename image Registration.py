

'''
step by step:
    1. Import 2 images
    2. Convert to gray scale
    3. Initiate ORB detector
    4. Find key points and describe them
    5. Match key points - Brute force matcher
    6. RANSAC (reject bad keypoints)
    7. Register two images (use homology)

'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

im1 = cv2.imread(r"C:\Users\Admin\Desktop\books3-1536x783.png")    # Image that needs to be registered.
im2 = cv2.imread(r"C:\Users\Admin\Desktop\books1.png") # trainImage

plt.imshow(im1)
plt.imshow(im2)


# ctvColor : تصویر های رنگی را به صورت خاکستری درمیاورد
img1 = cv2.cvtColor(src=im1, code=cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(src=im2, code=cv2.COLOR_BGR2GRAY)


# Initiate ORB detector
orb = cv2.ORB_create(nfeatures = 50)  #Registration works with at least 50 points


'''detectAndCompute:  نقاط کلیدی را استخراج می‌کند و آن‌ها را به همراه
 توصیفگرهایشان باز می‌گرداند.'''
# find the keypoints and descriptors with orb
kp1, des1 = orb.detectAndCompute(image = img1, mask = None)  #kp1 --> list of keypoints
kp2, des2 = orb.detectAndCompute(image = img2, mask = None)


#Brute-Force matcher takes the descriptor of one feature in first set and is 
#matched with all other features in second set using some distance calculation.
# create Matcher object
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Match descriptors.
matches = matcher.match(des1, des2, None)  #Creates a list of all matches, just like keypoints

matcher = sorted(matches, key=lambda x:x.distance)  #sorted by age


#Like we used cv2.drawKeypoints() to draw keypoints, 
#cv2.drawMatches() helps us to draw the matches. 
#https://docs.opencv.org/3.0-beta/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
# Draw first 10 matches.
img3 = cv2.drawMatches(im1,kp1, im2, kp2, matches[:10], None)

cv2.imshow("Matches image", img3)
cv2.waitKey(0)

plt.imshow(img3)



#Now let us use these key points to register two images. 
#Can be used for distortion correction or alignment
#For this task we will use homography. 
# https://docs.opencv.org/3.4.1/d9/dab/tutorial_homography.html

# Extract location of good matches.
# For this we will use RANSAC.
#RANSAC is abbreviation of RANdom SAmple Consensus, 
#in summary it can be considered as outlier rejection method for keypoints.
#http://eric-yuan.me/ransac/
#RANSAC needs all key points indexed, first set indexed to queryIdx
#Second set to #trainIdx. 

points1 = np.zeros((len(matches), 2), dtype=np.float32)  #Prints empty array of size equal to (matches, 2)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
   points1[i, :] = kp1[match.queryIdx].pt    #gives index of the descriptor in the list of query descriptors
   points2[i, :] = kp2[match.trainIdx].pt    #gives index of the descriptor in the list of train descriptors



#Now we have all good keypoints so we are ready for homography.   
# Find homography
#https://en.wikipedia.org/wiki/Homography_(computer_vision)
# RANSAC: detecting outlier points and keeping inlier points.
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 

  # Use homography
height, width, channels = im2.shape
im1Reg = cv2.warpPerspective(im1, h, (width, height))  #Applies a perspective transformation to an image.
   

print("Estimated homography : \n",  h)

img4 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:10], None)
cv2.imshow("Registered image", im1Reg)
cv2.waitKey(0)

plt.imshow(im1Reg)
plt.imshow(img4)









