import numpy as np
import json
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

class Camera:
    def __init__(self, camera_number, rotation_matrix, rgb_file, depth_file, location):
        self.camera_number = camera_number
        self.extrinsics = self.create_extrinsic_matrix(np.array(rotation_matrix), np.array(location).reshape(3, 1))
        self.rotation_matrix = np.array(rotation_matrix)
        self.rgb_file = rgb_file
        self.depth_file = depth_file
        self.location = np.array(location)
    
    def create_extrinsic_matrix(self, rotation_matrix, translation_vector):

        extrinsic_matrix = np.eye(4) 
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_vector.flatten()
        return extrinsic_matrix

class Cameras:
    def __init__(self, intrinsics, extrinsics, num_cameras):
        self.num_cameras = num_cameras
        self.cameras = self.get_cameras(extrinsics)
        self.intrinsics = self.get_intrinsics(intrinsics, homogeneous=False)

    def get_camera(self, camera_number):
        return self.cameras.get(camera_number, None)

    def get_intrinsics(self, path, homogeneous):
        with open(path, 'r') as file:
            camera_info = json.load(file)
        focal_length = camera_info["focal_length"]
        c_x, c_y = camera_info["c_x_c_y"]

        f_x = focal_length
        f_y = focal_length

        # Construct the intrinsic matrix
        K = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1]
        ])
        if homogeneous:
            intrinsic_matrix = np.eye(4)
            intrinsic_matrix[:3, :3] = K
            return intrinsic_matrix

        return K

    def get_cameras(self, path):
        with open(path, 'r') as file:
            cameras = json.load(file)
        
        data = {}
        for i in range(self.num_cameras):
            key = f"Camera_{i}"
            camera = cameras[key]
            location = camera["location"]
            rotation_matrix = camera["rotation_matrix"]
            rgb_file = f"Sofa_output3/rgb_camera_{i}.png"
            depth_file = f"Sofa_output3/depth_camera_{i}.png0001.png"
            data[i] = Camera(i, rotation_matrix, rgb_file, depth_file, location)
        return data


class Stereo:
    def __init__(self, camera1, camera2, intrinsics):
        self.camera1 = camera1
        self.camera2 = camera2
        self.intrinsics = intrinsics

    def findMatches(self, img1_path, img2_path):
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            

            sift = cv2.SIFT_create()

            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

    
            matched_points_img1 = np.float32([kp1[match.queryIdx].pt for match in matches])
            matched_points_img2 = np.float32([kp2[match.trainIdx].pt for match in matches])

            return (matched_points_img1, matched_points_img2)
    
    def triangulate(correspondences, P1, P2):
        pass

    def find_projection_matrices(self, intrinsics, cam1, cam2):
        cam1_location = cam1.location.reshape(3, 1)
        cam2_location = cam2.location.reshape(3, 1)
        P1 = intrinsics @ np.hstack((cam1.rotation_matrix, -cam1.rotation_matrix @ cam1_location))
        P2 = intrinsics @ np.hstack((cam2.rotation_matrix, -cam2.rotation_matrix @ cam2_location))

        return P1, P2
    
    def triangulate_points(self, points1, points2, P1, P2):
        points_4d_homogeneous = cv2.triangulatePoints(P1, P2, points1.T, points2.T)

        points_3d = points_4d_homogeneous[:3, :] / points_4d_homogeneous[3, :]
        return points_3d.T 
        
    def create_depth_map(self, img_shape, points2D, points3D):
        depth_map = np.zeros(img_shape, dtype=np.float32)

        for (point2D, point3D) in zip(points2D, points3D):
            x, y = int(point2D[0]), int(point2D[1])
            z = point3D[2]  
            if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:              
                depth_map[y, x] = z

        # Normalize the depth map 
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return depth_map_normalized
    
    def run(self):
        matches = self.findMatches(self.camera1.rgb_file, self.camera2.rgb_file)
        P1,P2 = self.find_projection_matrices(self.intrinsics, self.camera1, self.camera2)
        points_3D = self.triangulate_points(matches[0], matches[1], P1, P2)

        img = cv2.imread(self.camera1.rgb_file, cv2.IMREAD_GRAYSCALE)
        depth_map = self.create_depth_map(img.shape, matches[0], points_3D)

        cv2.imwrite("depth_map.png", depth_map)
        cv2.imshow("Depth Map", depth_map)







        


if __name__ == "__main__":
    num_cameras = 150
    intrinsic_path = "Sofa_output3/camera_intrinsics.json"
    extrinsic_path = "Sofa_output3/render_data.json"

    cameras = Cameras(intrinsic_path,extrinsic_path,num_cameras)
    camera0 = cameras.cameras[0]
    camera1 = cameras.cameras[1]
    stereo = Stereo(camera0, camera1, cameras.intrinsics)
    stereo.run()






