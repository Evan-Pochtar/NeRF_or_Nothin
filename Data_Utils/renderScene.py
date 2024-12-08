import bpy
import math
import os
import numpy as np
import json
import random
from mathutils import Vector
from mathutils import Matrix
output_dir = "/home/will/Documents/csci5561/scenes/lego_output_big/"
scene = bpy.context.scene
scene.render.resolution_x = 100
scene.render.resolution_y = 100

def look_at(camera, target):
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
def delete_all_cameras():
    """
    Deletes all objects in the current scene
    """
    deleteListObjects = ['CAMERA']
    bpy.ops.object.select_all(action='DESELECT')
    for o in bpy.context.scene.objects:
        for i in deleteListObjects:
            if o.type == i:
                o.select_set(True)

    bpy.ops.object.delete() 

def setup_cameras():
    delete_all_cameras()
    num_cameras = 300
    radius = 10
    for i in range(num_cameras):
        phi = math.acos(1 - 2 * random.random())
        theta = 2 * math.pi * random.random()
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        bpy.ops.object.camera_add(location=(x, y, z))
        camera = bpy.context.object

        camera.name = f"Camera_{i}"

        look_at(camera, Vector((0, 0, 0)))


def setup_compositor():
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links
    for node in tree.nodes:
        tree.nodes.remove(node)
    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    render_layers.location = (0, 0)

    map_range = tree.nodes.new(type="CompositorNodeMapRange")
    map_range.location = (200, 0)
    map_range.inputs[1].default_value = 0.1
    map_range.inputs[2].default_value = 10.0
    map_range.inputs[3].default_value = 0
    map_range.inputs[4].default_value = 1
    file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    file_output.location = (400, 0)
    file_output.format.file_format = "PNG"
    file_output.format.color_depth = "16"
    file_output.file_slots.clear()
    file_output.file_slots.new("DepthMap")

    links.new(render_layers.outputs["Depth"], map_range.inputs[0])
    links.new(map_range.outputs[0], file_output.inputs[0])

    return file_output
def orthogonalize(matrix):
    rotation = matrix.to_3x3()
    u, _, v = np.linalg.svd(np.array(rotation))
    ortho_rotation = Matrix(u @ v)
    return ortho_rotation
def listify_matrix(matrix):
    rotation = orthogonalize(matrix.to_3x3()).normalized()
    translation = matrix.to_translation()

    normalized_matrix = Matrix.Translation(translation) @ rotation.to_4x4()

    return [list(row) for row in normalized_matrix]

def render_rgb(scene, output_dir, cameras, render_data):
    import numpy as np
    from mathutils import Matrix

    os.makedirs(output_dir, exist_ok=True)

    for i, camera in enumerate(cameras):
        scene.camera = camera

        rgb_file = os.path.join(output_dir, f"rgb_camera_{i}.png")
        scene.render.filepath = rgb_file
        bpy.ops.render.render(write_still=True)
        render_data[camera.name] = {
                    "rgb_file": rgb_file,
                    "pose": listify_matrix(camera.matrix_world)
                }


def render_depth(scene, output_dir, cameras, render_data):
    file_output_node = setup_compositor()
    os.makedirs(output_dir, exist_ok=True)

    for i, camera in enumerate(cameras):
        scene.camera = camera

        file_output_node.base_path = output_dir
        file_output_node.file_slots[0].path = f"depth_camera_{i}.png"

        bpy.ops.render.render(write_still=False)

        depth_file = os.path.join(output_dir, f"depth_camera_{i}.png")
        render_data[camera.name]["depth_file"] = depth_file



def save_camera_intrinsics(output_file):
    cameras_intrinsics = {}
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            camera = obj.data
            resolution_x = bpy.context.scene.render.resolution_x
            resolution_y = bpy.context.scene.render.resolution_y
            focal_length = camera.lens
            sensor_width = camera.sensor_width
            sensor_height = camera.sensor_height
            c_x = resolution_x / 2
            c_y = resolution_y / 2
            pixel_aspect_ratio = (
                bpy.context.scene.render.pixel_aspect_y
                / bpy.context.scene.render.pixel_aspect_x
            )
            cameras_intrinsics[obj.name] = {
                "resolution": [resolution_x, resolution_y],
                "focal_length": focal_length,
                "sensor_size": [sensor_width, sensor_height],
                "c": [c_x, c_y],
                "aspect_ratio": pixel_aspect_ratio,
            }
    with open(output_file, "w") as f:
        json.dump(cameras_intrinsics, f, indent=4)
# setup_cameras()
os.makedirs(output_dir, exist_ok=True)


setup_cameras()
save_camera_intrinsics(output_dir + "camera_intrinsics.json")

render_data = {}

cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]

render_rgb(scene, output_dir, cameras, render_data)
render_depth(scene, output_dir, cameras, render_data)
render_data_file = os.path.join(output_dir, "render_data.json")
with open(render_data_file, "w") as f:
    json.dump(render_data, f, indent=4)

print(f"Render data with camera poses saved to: {render_data_file}")
