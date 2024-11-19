import bpy
import math
import os
import numpy as np
import json
import random
from mathutils import Vector

output_dir = "/home/will/Documents/csci5561/blender/scenes/Sofa/Sofa_output12/"
scene = bpy.context.scene

# https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
def look_at(camera, target):
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
def get_cameras():
    out = []
    for i in bpy.data.objects:
        if i.type == "CAMERA":
            out.append(i)
    return out
def setup_cameras():
    num_cameras = 5
    radius = 10
    for i in range(num_cameras):
        # https://blender.stackexchange.com/questions/151319/adding-camera-to-scene
        angle = (2 * math.pi / num_cameras) * i
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = random.uniform(-4, 8)
        bpy.ops.object.camera_add(location=(x, y, z))
        camera = bpy.context.object
        camera.name = "c_"+str(i)
        look_at(camera, Vector((0, 0, 0)))
# https://stackoverflow.com/questions/14982836/rendering-and-saving-images-through-blender-python
def render_rgb(scene, output_dir, render_data):
    cameras = get_cameras()
    os.makedirs(output_dir, exist_ok=True)
    for i, camera in enumerate(cameras):
        scene.camera = camera
        rgb_file = os.path.join(output_dir, f"rgb_camera_{i}.png")
        scene.render.filepath = rgb_file
        bpy.ops.render.render(write_still=True)
        render_data[camera.name] = {
            "rgb_file": rgb_file,
            "location": list(camera.location),
            # https://blender.stackexchange.com/questions/214567/how-to-use-the-rotation-matrix-calculated-by-matlab-in-blender
            "rotation_matrix": [list(row) for row in camera.matrix_world.to_3x3()],
        }


def render_depth(scene, output_dir, render_data):
    cameras = get_cameras()
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links
    for node in tree.nodes:
        tree.nodes.remove(node)
    # https://blender.stackexchange.com/questions/42579/render-depth-map-to-image-with-python-script
    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    render_layers.location = (0, 0)
    # https://docs.blender.org/api/current/bpy.types.CompositorNodeNormalize.html
    normalize = tree.nodes.new(type="CompositorNodeNormalize")
    normalize.location = (200, 0)

    file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    file_output.location = (400, 0)
    file_output.format.file_format = "PNG"
    file_output.file_slots.clear()
    file_output.file_slots.new("DepthMap")
    links.new(render_layers.outputs["Depth"], normalize.inputs[0])
    links.new(normalize.outputs[0], file_output.inputs[0])

    for i, camera in enumerate(cameras):
        scene.camera = camera
        file_output.base_path = output_dir
        file_output.file_slots[0].path = f"depth_camera_{i}.png"
        bpy.ops.render.render(write_still=False)
        depth_file = os.path.join(output_dir, f"depth_camera_{i}.png")
        if camera.name not in render_data:
            render_data[camera.name] = {}
        render_data[camera.name]["depth_file"] = depth_file


def save_intrinsics():
    # https://blender.stackexchange.com/questions/98546/set-render-camera-resolution-to-render-border-coordinates
    resolution_x = bpy.context.scene.render.resolution_x
    resolution_y = bpy.context.scene.render.resolution_y
    cam = None
    for i in bpy.data.objects:
        if i.type=="CAMERA":
            cam = i
            break
    # https://blender.stackexchange.com/questions/14745/how-do-i-change-the-focal-length-of-a-camera-with-python
    camera_data = cam.data
    focal_length = camera_data.lens
    c_x = resolution_x / 2
    c_y = resolution_y / 2
    intrinsics = {
        "focal_length": focal_length,
        "resolution_x": 1920,
        "resolution_y":1080,
        "c_x":c_x,
        "c_y":c_y
    }
    with open(output_dir + "camera_intrinsics.json", "w") as f:
        json.dump(intrinsics, f)

    print("saved intrinsics")



setup_cameras()
save_intrinsics()
render_data = {}
render_rgb(scene, output_dir, render_data)
render_depth(scene, output_dir, render_data)
render_data_file = os.path.join(output_dir, "render_data.json")
with open(render_data_file, "w") as f:
    json.dump(render_data, f)
print("Render data with camera poses saved to: " + render_data_file)
