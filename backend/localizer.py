import os
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt 
from PIL import Image

class Localizer:
    def __init__(self):
        self.initial_postion = np.eye(4)
       
        self.params = [316.36175731, 0, 313.10637575, 0, 0, 315.84770187, 233.20697605, 0, 0, 0,1,0]
        self.params = np.array(self.params)
        self.P = np.reshape(self.params, (3,4))
        self.K = self.P[0:3, 0:3]
        self.last_image = None
        self.last_pos = None


        self.orb = cv2.xfeatures2d.SIFT_create(1000)
        # self.orb = cv2.xfeatures2d.SURF_create(1000)
        # sift = cv2.xfeatures2d.SIFT_create(100)
        # self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=80)
        self.flann = cv2.BFMatcher()
        # self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        return
    

    def get_match(self, last_image, current_image):
        crop_height = 380
        last_image = last_image[:crop_height,:]
        current_image = current_image[:crop_height,:]
        keypoints1, descriptors1 = self.orb.detectAndCompute(last_image, None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(current_image, None)
        
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        # store all the good matches as per Lowe's ratio test.
        
        good = []
        for i in range(len(matches)):
            if len(matches[i]) == 0:
                continue
            elif len(matches[i]) > 1:
                m = matches[i][0]
                n = matches[i][1]
                if m.distance < 0.8*n.distance:
                    good.append(m)
            else:
                m = matches[i][0]
                good.append(m)

        q1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ])
        q2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ])


        draw_params = dict(matchColor = -1, # draw matches in green color
                singlePointColor = None,
                matchesMask = None, # draw only inliers
                flags = 2)
        img3 = cv2.drawMatches(current_image, keypoints1, last_image, keypoints2, good ,None,**draw_params)
        cv2.imshow("image", img3)
        cv2.waitKey(0)
        return q1, q2
    
    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def decomp_essential_mat(self, E, q1, q2):
        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1,np.ndarray.flatten(t))
        T2 = self._form_transf(R2,np.ndarray.flatten(t))
        T3 = self._form_transf(R1,np.ndarray.flatten(-t))
        T4 = self._form_transf(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        K = np.concatenate(( self.K, np.zeros((3,1)) ), axis = 1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                        np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)
            

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives)
        if (max == 2):
            # print(-t)
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            # print(-t)
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t)


    def get_post(self, q1, q2):
        Essential, mask = cv2.findEssentialMat(q1, q2, self.K)
        # print ("\nEssential matrix:\n" + str(Essential))
        R, t = self.decomp_essential_mat(Essential, q1, q2)
        return self._form_transf(R,t)
    

    def localize(self, image: Image):
        """
        return current location
        return fomrat: (x, y) in centimeters and top-left is (0, 0)
        """
        image = np.array(image)[:, :]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.last_image is None:
            self.last_image = image
            self.last_pos = self.initial_postion
            x = self.last_pos[0, 3]
            y = self.last_pos[2, 3]
            return x, y
        
        q1, q2 = self.get_match(self.last_image, image)
        transf = self.get_post(q1, q2)
        cur_pose = np.matmul(self.last_pos, np.linalg.inv(transf))
        x = cur_pose[0, 3]
        y = cur_pose[2, 3]
        self.last_image = image
        self.last_pos = cur_pose
        return x, y
    

if __name__ == "__main__":
    x_ = []
    y_ = []
    loc = Localizer()
    image_path = "KITTI_sequence_1/image_l"
    image_paths = [os.path.join(image_path, file) for file in sorted(os.listdir(image_path)) if file.endswith("png")]
    for image in image_paths:
        image = Image.open(image)
        x, y = loc.localize(image)
        print(x, y)
        x_.append(x)
        y_.append(y)
    plt.plot(x_, y_, marker='o')
    plt.show()

