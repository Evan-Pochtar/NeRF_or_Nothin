import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Concatenate  # Import Concatenate layer
from tensorflow.keras.models import load_model
from tqdm import tqdm
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as colors

# Ensure the 'materials' directory exists
os.makedirs('materials', exist_ok=True)

@tf.function  # Optimize download for TensorFlow's eager execution
def download_data():
    if not os.path.exists('tiny_nerf_data.npz'):
        os.system('wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz')

@tf.function
def rgb_to_grayscale(rgb_image):
    return tf.image.rgb_to_grayscale(rgb_image)

@tf.function
def posenc(x, L_embed=6):
    rets = [x]
    for i in range(L_embed):
        scale = 2.0**i
        rets.extend([tf.sin(scale * x), tf.cos(scale * x)])
    return tf.concat(rets, -1)

def init_model(D=8, W=256, L_embed=6):
    relu = tf.keras.layers.ReLU()
    dense = lambda W=W, act=relu: tf.keras.layers.Dense(W, activation=act)

    # Fixed shape definition
    inputs = tf.keras.Input(shape=(3 + 3 * 2 * L_embed,))
    outputs = inputs
    for i in range(D):
        outputs = dense()(outputs)
        if i % 4 == 0 and i > 0:
            # Use Concatenate for KerasTensor
            outputs = Concatenate()([outputs, inputs])
    outputs = dense(1, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

@tf.function
def get_rays(H, W, focal, c2w):
    # Ensure consistent data types
    H = tf.cast(H, tf.float32)
    W = tf.cast(W, tf.float32)
    focal = tf.cast(focal, tf.float32)

    i, j = tf.meshgrid(
        tf.range(W, dtype=tf.float32),
        tf.range(H, dtype=tf.float32),
        indexing='xy'
    )
    dirs = tf.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


@tf.function
def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False, embed_fn=posenc):
    z_vals = tf.linspace(near, far, N_samples)
    if rand:
        z_vals += tf.random.uniform(tf.shape(z_vals), -1e-4, 1e-4)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    pts_flat = tf.reshape(pts, [-1, 3])
    pts_flat = embed_fn(pts_flat)
    raw = network_fn(pts_flat)
    raw = tf.reshape(raw, tf.concat([tf.shape(pts)[:-1], [1]], axis=0))

    sigma_a = tf.nn.relu(raw)
    sigma_a = tf.squeeze(sigma_a, axis=-1)

    dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1], tf.broadcast_to([1e10], z_vals[..., :1].shape)], -1)
    alpha = 1.0 - tf.exp(-sigma_a * dists)
    weights = alpha * tf.math.cumprod(1.0 - alpha + 1e-10, -1, exclusive=True)

    gray_map = tf.reduce_sum(weights[..., None] * raw, -2)
    depth_map = tf.reduce_sum(weights * z_vals, -1)
    acc_map = tf.reduce_sum(weights, -1)

    return gray_map, depth_map, acc_map

def train_nerf(images, poses, H, W, focal, N_samples=64, N_iters=1000, i_plot=25):
    model = init_model()
    optimizer = tf.keras.optimizers.Adam(5e-4)

    psnrs = []
    iternums = []

    # Select the last image and pose for testing
    testimg, testpose = images[-1], poses[-1]  # Fix index here

    t = time.time()
    for i in range(N_iters + 1):
        img_i = np.random.randint(len(images) - 1)  # Exclude test image
        target = images[img_i]
        pose = poses[img_i]
        rays_o, rays_d = get_rays(H, W, focal, pose)

        with tf.GradientTape() as tape:
            grey_map, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples, rand=True)
            loss = tf.reduce_mean(tf.square(grey_map - target))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if i % i_plot == 0:
            print(i, (time.time() - t) / i_plot, 'secs per iter')
            t = time.time()

            rays_o, rays_d = get_rays(H, W, focal, testpose)
            grey_map, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
            loss = tf.reduce_mean(tf.square(grey_map - testimg))
            psnr = -10. * tf.math.log(loss) / tf.math.log(10.)

            psnrs.append(psnr.numpy())
            iternums.append(i)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(grey_map)
            plt.title(f'Iteration: {i}')
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title('PSNR')
            plt.savefig(f'materials/training_progress_{i}.png')
            plt.close()
            
    model.save("materials/nerf_model.keras")
    print("Model saved to materials/nerf_model")

    return model


def create_pose_transforms():
    # Transformation matrices
    trans_t = lambda t : tf.convert_to_tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1],
    ], dtype=tf.float32)

    rot_phi = lambda phi : tf.convert_to_tensor([
        [1,0,0,0],
        [0,tf.cos(phi),-tf.sin(phi),0],
        [0,tf.sin(phi), tf.cos(phi),0],
        [0,0,0,1],
    ], dtype=tf.float32)

    rot_theta = lambda th : tf.convert_to_tensor([
        [tf.cos(th),0,-tf.sin(th),0],
        [0,1,0,0],
        [tf.sin(th),0, tf.cos(th),0],
        [0,0,0,1],
    ], dtype=tf.float32)

    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w

    return pose_spherical

def render_and_save_novel_views_interactive(model, H, W, focal, N_samples=64):
    pose_spherical = create_pose_transforms()
    
    # Define view parameters
    view_params = [
        (100, -30, 4),   # Example view 1
        (200, -60, 4.5), # Example view 2
        (300, -45, 3.5)  # Example view 3
    ]
    
    for idx, (theta, phi, radius) in enumerate(view_params):
        c2w = pose_spherical(theta, phi, radius)
        rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
        
        # Render image and depth map
        grey_map, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        
        # Normalize depth for visualization
        depth_normalized = (depth - tf.reduce_min(depth)) / (tf.reduce_max(depth) - tf.reduce_min(depth))
        
        # Convert normalized depth to RGB using a colormap
        colormap = cm.get_cmap('viridis')  # Use 'viridis' colormap
        depth_rgb = (colormap(depth_normalized.numpy())[:, :, :3] * 255).astype(np.uint8)  # Extract RGB values
        
        # Create an interactive plot using Plotly
        fig = go.Figure(data=go.Surface(
            z=depth_normalized.numpy(),
            surfacecolor=depth_rgb,
            colorscale='Viridis',
            cmin=0,
            cmax=1,
        ))
        fig.update_layout(
            title=f'Interactive Depth Map (θ={theta}, φ={phi}, r={radius})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Depth',
            )
        )
        
        # Save the figure as an HTML file
        fig.write_html(f'materials/interactive_depth_map_{idx}.html')
        print(f"Saved interactive depth map to materials/interactive_depth_map_{idx}.html")


def main():
    download_data()
    
    # Load and preprocess data
    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    H, W = images.shape[1:3]
    
    # Check if model exists; if so, load it
    model_path = "materials/nerf_model"
    if os.path.exists(model_path):
        print("Loading saved model...")
        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")
    else:
        # Train the NeRF model
        model = train_nerf(images[:100], poses[:100], H, W, focal)
    
    # Render and save novel views
    render_and_save_novel_views_interactive(model, H, W, focal)

if __name__ == "__main__":
    main()