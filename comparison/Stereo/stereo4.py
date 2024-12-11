import numpy as np
import cv2
import json
import matplotlib.pyplot as plt


class Camera:
    def __init__(self, camera_number, pose, rgb_files):
        self.camera_number = camera_number
        self.extrinsic_matrix = pose
        self.rgb_files = rgb_files 

class Stereo:
    def __init__(self, cameras, intrinsics, extrinsics):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.cameras = cameras
        self.points = None

    def findMatches(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        sift = cv2.SIFT_create()

        #Find correspondences
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        matched_points_img1 = np.float32([kp1[match.queryIdx].pt for match in matches])
        matched_points_img2 = np.float32([kp2[match.trainIdx].pt for match in matches])

        return matched_points_img1, matched_points_img2

    def find_projection_matrices(self, intrinsics, cam1, cam2):

        intrinsics = np.array(intrinsics)
        cam1 = np.array(cam1.extrinsic_matrix)
        cam2 = np.array(cam2.extrinsic_matrix)

        # Compute projection matrices
        P1 = intrinsics @ cam1[:3, :]  
        P2 = intrinsics @ cam2[:3, :] 

        return P1, P2

    def triangulate_points(self, points1, points2, P1, P2):

        points_4d_homogeneous = cv2.triangulatePoints(P1, P2, points1.T, points2.T)

        points_3d = points_4d_homogeneous[:3, :] / points_4d_homogeneous[3, :]

        return points_3d
    
    def filter_points(self, main, points):

        points = self.points.T

        extrinsic_matrix = np.array(main.extrinsic_matrix)


        points_in_front = []

        for point in points:

            point = np.array([point[0], point[1], point[2], 1.0])  

            # Transform the point from world coordinates to camera coordinates 
            point_camera = np.dot(extrinsic_matrix, point) 


            # If the z-coordinate in camera coordinates is positive the point is in front
            if point_camera[2] > 0:
                points_in_front.append(point[:3]) # append the point

        return np.array(points_in_front)

    def create_depth_map_with_inverted_colormap(self, camera, points):
        #Create a depth map where close objects are dark and far objects are bright.
        if points is None or points.size == 0:
            print("No points to create depth map.")
            return
        
        points_3d = np.array(points)

        main_intrinsics = np.array(self.intrinsics)
        main_extrinsics = np.array(camera.extrinsic_matrix)
        
        #create projection matrix of the camera
        projection_matrix = main_intrinsics @ main_extrinsics[:3, :]

        points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        projected_points = projection_matrix @ points_homogeneous.T

        # Convert projected points to pixel
        u = projected_points[0, :] / projected_points[2, :]
        v = projected_points[1, :] / projected_points[2, :]
        depth = projected_points[2, :]  # Z-coordinate in camera space


        image_height = 100  
        image_width = 100   
        depth_map = np.full((image_height, image_width), np.inf) 

        #use a z_buffer to keep track of nearest point
        z_buffer = np.full((image_height, image_width), np.inf) 


        for i in range(len(u)):
            px, py = int(round(u[i])), int(round(v[i]))
            if 0 <= px < image_width and 0 <= py < image_height:
                # Only update if this point is closer than the current point in the Z-buffer
                if depth[i] < z_buffer[py, px]:
                    depth_map[py, px] = depth[i]
                    z_buffer[py, px] = depth[i]

        # Normalize the depth map
        finite_depths = depth_map[np.isfinite(depth_map)]  
        if len(finite_depths) > 0:
            min_depth = np.min(finite_depths)
            max_depth = np.max(finite_depths)
            depth_map_normalized = (depth_map - min_depth) / (max_depth - min_depth)
            depth_map_inverted_colormap = depth_map_normalized
        else:
            depth_map_normalized = depth_map  
            depth_map_inverted_colormap = depth_map 

        # Save the map
        depth_map_inverted_colormap = np.uint8(depth_map_inverted_colormap) 
        cv2.imwrite(f'StereoDepth{camera.camera_number}.png', depth_map_inverted_colormap)
  






    


    def run(self):

        # Loop through all image pairs
        for idx1, img1_path in enumerate(self.cameras):
            for idx2, img2_path in enumerate(self.cameras[idx1+1:], start=idx1+1):
                print(f"Processing Camera 1 Image {idx1+1} and Camera 2 Image {idx2+1}...")

                # Find matching points between Camera 1 and Camera 2
                matches = self.findMatches(img1_path, img2_path)

                camera1 = Camera(idx1, self.extrinsics[f"Camera_{idx1}"]["pose"], self.cameras[idx1])
                camera2 = Camera(idx2, self.extrinsics[f"Camera_{idx2}"]["pose"], self.cameras[idx2])


                # Get the projection matrices for both cameras
                P1, P2 = self.find_projection_matrices(self.intrinsics, camera1, camera2)

                # Triangulate the matching points to get 3D coordinates
                points_3D = self.triangulate_points(matches[0], matches[1], P1, P2)

                #Add points to the array
                if self.points is None or self.points.size == 0:
                    self.points = points_3D
                else:
                    self.points = np.concatenate((self.points, points_3D), axis=1)


        #for each camera create a depth map
        for i in range(300):
            maincam = Camera(i, self.extrinsics[f"Camera_{i}"]["pose"], self.cameras[i])
            points = self.filter_points(maincam,self.points)
            print(points.shape)
            self.create_depth_map_with_inverted_colormap(maincam, points)


        

        

if __name__ == "__main__":

    extrinsic_path = "path_to_extrinsic"
    
    # Camera 1 images
    camera1_images = [
        f"path_to_data{i}.png"
        for i in range(0, 300)
    ]


    intrinsic_matrix = np.array([
        [138.88888889, 0.0, 50.0],
        [0.0, 138.88888889, 50.0],
        [0.0, 0.0, 1.0]
    ])

    # Load extrinsics
    with open(extrinsic_path, 'r') as f:
        extrinsics = json.load(f)


    stereo = Stereo(camera1_images,intrinsic_matrix, extrinsics)
    stereo.run()
