import numpy as np
import cv2
from matplotlib import pyplot as plt
 
def extract_orb(img):
    """
    Extract orb feature from image
    :param img: input image
    :return:
    """
    # convert to gray
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
 
    # orb keypoint detector and descriptor
    orb_obj = cv2.ORB_create(nfeatures=1500)
 
    # extract orb keypoints and descriptors
    kp, desc = orb_obj.detectAndCompute(img_gray, None)
 
    # keypoints coordinate in 2D image
    kp = np.array([p.pt for p in kp]).T
    return kp, desc
 
 
def match_orb(descriptor_source, descriptor_target):
    """
    math keypoints of two images
    :param descriptor_source:
    :param descriptor_target:
    :return:
        
        تابع match_orb توصیفگرهای نقاط کلیدی دو تصویر مختلف را می‌گیرد و با استفاده
        از KNN دو نقطه کلیدی نزدیک به  هر نقطه کلیدی را پیدا می‌کند. در صورتی که فاصله
        توصیفگر نقطه کلیدی تا نقطه کلیدی اول به طرز قابل قبولی کمتر از فاصله نقطه کلیدی
        تا نقطه کلیدی دوم باشد، نشان می‌دهند که نقطه اول یک نقطه کلیدی متناظر خوب
        برای نقطه کلیدی است. در غیر این صورت از آن صرف نظر می‌شود و گفته می‌شود 
        تطبیقی برای آن نقطه پیدا نشده‌است. خروجی این تابع اندیس نقاط کلیدی منطبق شده است.
        
    """
    # Matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor_source, descriptor_target, k=2)
 
    # keep good matches
    good_matches = np.array([], dtype=np.int32).reshape((0, 2))
    matches_num = len(matches)
    for i in range(matches_num):
        if matches[i][0].distance <= 0.8 * matches[i][1].distance:
            temp = np.array([matches[i][0].queryIdx,
                             matches[i][0].trainIdx])
            good_matches = np.vstack((good_matches, temp))
    return good_matches
 
 
def estimate_affine_transform(src_points, trg_points):
    """
    Estimate affine transform by getting a set of points and their transformed
    :param src_points:
    :param trg_points:
    :return:
        
        فرض کنید که نقاط A را داریم و نقاط B نقاط متناظر با نقاط A در یک دستگاه مختصات
        دیگر هستند که با استفاده از نگاشت X[A]+Y = B به دست می‌آیند. حالا فرض کنید که 
        نقاط A و نقاط B را داده‌اند و می‌خواهیم این نگاشت خطی را پیدا کنیم. برای این کار
        از تابع estimate_affine_transform استفاده می‌کنیم.
        
    """
    num_of_points = src_points.shape[1]
    M = np.zeros((2 * num_of_points, 6))
    for i in range(num_of_points):
        temp = [[src_points[0, i], src_points[1, i], 0, 0, 1, 0],
                [0, 0, src_points[0, i], src_points[1, i], 0, 1]]
        M[2 * i: 2 * i + 2, :] = np.array(temp)
    b = trg_points.T.reshape((2 * num_of_points, 1))
    theta = np.linalg.lstsq(M, b)[0]
    X_trans = theta[:4].reshape((2, 2))
    Y_trans = theta[4:]
    return X_trans, Y_trans
 
 
def mse_error(X_trans, Y_trans, src_points, trg_points):
    """
 
    :param X_trans: transform parameter X[]+Y
    :param Y_trans: transform parameter X[]+Y
    :param src_points: source points
    :param trg_points: traget points
    :return:
        
        فرض کنید که نقاط A را در یک دستگاه مختصات داریم و می‌دانیم که اگر آن‌ها را با نگاشت
        خطی X[A] + ‌Y به یک دستگاه مختصات دیگر ببریم، نقطه متناظر نقاط A، نقاط B می‌شود.
        حالا فرض کنید که A و B را داریم ولی نگاشت خطی X[ ]+Y را نداریم. به ما یک نگاشت 
        خطی P[ ] + Q می‌دهند و می‌خواهیم بدانیم که چقدر این  نگاشت خطی دقیق است و عمل
        نگاشت را درست انجام می‌دهد. برای این کار نقاط A را با استفاده از PA+Q به مختصات
        جدید می‌بریم و مجذور مربعات فاصله نقاط را با نقاط متناظرشان که B باشد می‌سنجیم.
        هر چه این مقدار کمتر باشد نشان می‌دهد نگاشت خطی داده شده بهتر است. این کاری است
        که تابع mse_error انجام می‌دهد.

    """
    # transform source points by affine transfor X[]+Y
    src_map_on_trg = np.dot(X_trans, src_points) + Y_trans
 
    # calculate MSE error
    diff_square = np.power(src_map_on_trg - trg_points, 2)
    mse = np.sqrt(np.sum(diff_square, axis=0))
 
    # return mse error
    return mse
 
 
def ransac(points_src, points_target):
    """
 
    :param points_src:
    :param points_target:
    :return:
        
        همه نقاط متناظری که با استفاده از تطبیق نقاط کلیدی با استفاده از توصیفگر‌هایشان
        به دست آمد، تقاط متناظر خوبی نیستند و اشتباهاتی در آن‌ها موجود است. تابع RANSAC 
        هر بار تعداد از نقاط را انتخاب می‌کند. سپس با استفاده از آن نقاط و 
        تابع estimate_affine_transform یک نگاشت خطی پیدا می‌کند و سپس با استفاده mse_error دقت
        نگاشت به دست آمده برای سایر نقاط را می‌سنجد. اینگونه بهترین نگاشت را پیدا می‌کند
        و تناظر‌های خوب و بد را از هم تشخیص می‌دهد.
        
    """
    # in each iteration select k point randomly
    K = 3
    # if mse error in less than threshold consider it
    mse_threshold = 1.0
    # maximum number of iterations
    number_of_iteration = 3000
 
    inliers_num_final = 0
    X_trans_final = None
    Y_trans_final = None
    inliers = None
 
    # for number of iteration repeat
    for i in range(number_of_iteration):
 
        # select K points randomly
        k_selected = np.random.randint(0, points_src.shape[1], (K, 1))
 
        # find affine transform by K selected points
        X_trans, Y_trans = estimate_affine_transform(src_points=points_src[:, k_selected],
                                                     trg_points=points_target[:, k_selected])
 
        # calculate mse error of affine transform
        mse = mse_error(X_trans=X_trans, Y_trans=Y_trans,
                        src_points=points_src, trg_points=points_target)
 
        if not (mse is None):
            inliers_tmp = np.where(mse < mse_threshold)
            inliers_num_tmp = len(inliers_tmp[0])
 
            # keep best affine transform until now
            if inliers_num_tmp > inliers_num_final:
                inliers_num_final = inliers_num_tmp
                inliers = inliers_tmp
                X_trans_final = X_trans
                Y_trans_final = Y_trans
        else:
            pass
 
    # return affine transform X[]+Y and inliers
    return X_trans_final, Y_trans_final, inliers
 
 
def affine_matrix(points_in_src_img, points_in_trg_img, good_matches):
    """
    Calculate affine transform
    :param points_in_src_img: 
    :param points_in_trg_img: 
    :param good_matches: 
    :return: 
        
        تابع affine_matrix نقاط کلیدی متناظر که توسط match_orb پیدا شده‌اند را می‌گیرد.
        سپس با استفاده از ransac تناظر‌های خوب از میان آن‌ها را پیدا می‌کند و تناظر‌های بد
        را کنار می‌گذارد. سپس با استفاده از تناظر‌های خوب پیدا شده توسط Ransac  و
        تابع estimate_affine_transform، ماتریس نگاشت خطی M را پیدا می‌کند.
        
    """
    # keypoints in image 1 that are in matched points
    points_in_src_img = points_in_src_img[:, good_matches[:, 0]]
 
    # keypoints in image 2 that are in matched points
    points_in_trg_img = points_in_trg_img[:, good_matches[:, 1]]
 
    # calculate inliers with RANSAC algorithm
    _, _, inliers = ransac(points_in_src_img, points_in_trg_img)
 
    # keypoints in image 1 that are in inliers points
    points_in_src_img = points_in_src_img[:, inliers[0]]
    # keypoints in image 2 that are in inliers points
    points_in_trg_img = points_in_trg_img[:, inliers[0]]
 
    # estimate affine transform X[]+Y by inliers points in first and second image
    X_trans, Y_trans = estimate_affine_transform(points_in_src_img, points_in_trg_img)
    M = np.hstack((X_trans, Y_trans))
    return M
 
def two_image_registraion_without_padding(img_source, img_target):
    """
    Gets two image an do image registration
    :param img_source:
    :param img_target:
    :return:
        
 تابع two_image_registraion_without_padding، دو تصویر را می‌گیرد و با استفاده 
 از تابع‌هایی که گفته شده کار ثبت تصویر یا image registration را انجام می‌دهد.
 فقط مشکلی که این تابع دارد این است که بعد از انتقال تصویر اول به دستگاه 
 مختصات تصویر دوم، ممکن است بخشی از تصویر اول بیرون بزند. این مشکل در 
 تابع two_images_registraion_padding حل شده است.    

    """
    # extract orb features for both images
    keypoint_source, descriptor_source = extract_orb(img_source)
    keypoint_target, descriptor_target = extract_orb(img_target)
 
    # math keypoints of two image
    matched_points = match_orb(descriptor_source, descriptor_target)
 
    # calculate affine matrix
    M = affine_matrix(keypoint_source, keypoint_target, matched_points)
    rows, cols, _ = img_target.shape
 
    # transform source image by H
    transformed_src_img = cv2.warpAffine(src=img_source, M=M, dsize=(cols, rows))
 
    # find where will be corners of source image
    new_corners = np.dot(a=M, b=np.array([[0, img_source.shape[1]-1, 0, img_source.shape[1]-1],
                                          [0, 0, img_source.shape[0]-1, img_source.shape[0]-1],
                                          [1, 1, 1, 1]]))
 
    # merge two images
    merged_image = cv2.addWeighted(src1=transformed_src_img, alpha=0.5, src2=img_target, beta=0.5, gamma=0)
 
    return merged_image, new_corners


def two_images_registraion_padding(img_source, img_target):
    """
    Does image registration with padding. no part of imaged will be lost
    :param img_source:
    :param img_target:
    :return:
       
    """
    # do image registration without padding
    merged_image_without_padding, new_corners = two_image_registraion_without_padding(img_source=img_source,
                                                                                      img_target=img_target)
 
    # find who two padd
    x_min = int(np.floor(new_corners[0, :].min()))
    x_max = int(np.ceil(new_corners[0, :].max()))
    y_min = int(np.floor(new_corners[1, :].min()))
    y_max = int(np.ceil(new_corners[1, :].max()))
 
    left, right, top, bottom = 0, 0, 0, 0
 
    if x_min < 0:
        left = -x_min
 
    if x_max > img_target.shape[1]:
        right = x_max-img_target.shape[1]
 
    if y_min < 0:
        top = -y_min
 
    if y_max > img_target.shape[0]:
        bottom = y_max - img_target.shape[0]
 
    # pad target image
    padded_img_target = cv2.copyMakeBorder(src=img_target, top=top, bottom=bottom,
                                           left=left, right=right,
                                           borderType=cv2.BORDER_CONSTANT)
 
    # do image registration
    merged_image_with_padding, _ = two_image_registraion_without_padding(img_source=img_source,
                                                                         img_target=padded_img_target)
 
    return merged_image_with_padding
 
 
 

     # ================================================
    # test image registration with two image
    # read images
    img_source = cv2.imread(r"C:\Users\Admin\Desktop\785793b8-7b8e-4716-80d1-e6e9c10b429f.png")
    img_target = cv2.imread(r"C:\Users\Admin\Desktop\70f4f74a-c4a2-49c2-b766-6b168ba73ad0.png")

    # do image registration
    merged_image = two_images_registraion_padding(img_source=img_source, img_target=img_target)
 
    # save image
    cv2.imwrite(filename='result_of_registration_two_images_tehran_map.png', img=merged_image)
 
    # cv2.imshow('img_source', img_source)
    # cv2.imshow('img_target', img_target)
    cv2.imshow('result_of_registration_two_images_tehran_map', merged_image)
    plt.imshow(merged_image)
    merged_image.save(r"C:\Users\Admin\Desktop\merged_image.jpg")
