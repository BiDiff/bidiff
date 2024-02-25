import torch
import numpy as np
import os
import tqdm

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)

root = '/home/lihe/dataset/ShapeNet'
cat_id = '03001627'
out_dir = '/home/lihe/dataset/latents'
os.makedirs(out_dir, exist_ok=True)
data_dir = os.path.join(root, cat_id)

obj_id_list = os.listdir(data_dir)

if __name__ == '__main__':
    for obj_id in obj_id_list:
        obj_path = os.path.join(data_dir, obj_id, 'model.obj')
        # obj_path = "/home/lihe/dataset/ShapeNet/03001627/98547d9147a58195f51f77a6d7299806/model.obj"
        # obj_path = "/home/lihe/dataset/ShapeNet/03001627/971539fb57b476d5c40593250b73d0c7/model.obj"
        # obj_path = "./test_model2.obj"
        # This may take a few minutes, since it requires rendering the model twice
        # in two different modes.
        batch = load_or_create_multimodal_batch(
            device,
            model_path=obj_path,
            mv_light_mode="basic",
            mv_image_size=256,
            # cache_dir="example_data/cactus/cached",
            # cache_dir="./cached",
            cache_dir=None,
            verbose=True, # this will show Blender output during renders
        )
        with torch.no_grad():
            latent = xm.encoder.encode_to_bottleneck(batch)
            save_dir = os.path.join(out_dir, obj_id)
            os.makedirs(save_dir, exist_ok=True)
            if not os.path.exists(os.path.join(save_dir, 'latent.npy')):
                np.save(os.path.join(save_dir, 'latent.npy'), latent.detach().cpu().numpy())
            else:
                print(f"==={obj_id} has been processed====, skipping...")
                continue
            
            # check rendering
            # render_mode = 'stf' # you can change this to 'nerf'
            render_debug = False
            if render_debug:
                render_mode = 'nerf' # you can change this to 'nerf'
                # size = 128 # recommended that you lower resolution when using nerf
                size = 64 # recommended that you lower resolution when using nerf

                cameras = create_pan_cameras(size, device)

                images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
                # print(len(images))
                images[0].save('test_dataset.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
                # gif_widget(images)
                # display(gif_widget(images))
                print(f"===DONE {obj_id}===")
                exit()
            # exit()