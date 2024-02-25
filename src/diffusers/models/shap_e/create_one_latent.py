import torch
import numpy as np
import os
import tqdm

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)


if __name__ == '__main__':
    # obj_path = "/home/lihe/dataset/ShapeNet/03001627/98547d9147a58195f51f77a6d7299806/model.obj"
    # obj_path = "debug_data/bruni.glb"
    # obj_path = "debug_data/eagle.glb"
    # obj_path = "debug_data/fly_eagle.glb"
    # obj_path = "debug_data/eagle_head.glb"
    # obj_path = "debug_data/dodo.glb"
    # obj_path = "debug_data/dragon.glb"
    # obj_path = "debug_data/robot_shark.glb"
    # obj_path = "debug_data/space_ship.glb"
    # obj_path = "debug_data/china_dragon.glb"
    # obj_path = "debug_data/toy_rocket.glb"
    # obj_path = "debug_data/sample_4.ply"
    # obj_path = "debug_data/sample_56_2.ply"
    # obj_path = "debug_data/sample_45_2.ply"
    # obj_path = "debug_data/sample_114.ply"
    # obj_path = "debug_data/sample_41_2.ply"
    # obj_path = "debug_data/sample_45_2.ply"
    # obj_path = "debug_data/chinese_tower.glb"
    # obj_path = "debug_data/chinese_tower2.glb"
    # obj_path = "debug_data/gothic_tower.glb"
    # obj_path = "debug_data/super_car.glb"
    # obj_path = "debug_data/cow_elephant.glb"
    # obj_path = "debug_data/vango_house.glb"
    # obj_path = "debug_data/cow.glb"
    # obj_path = "debug_data/teapot.glb"
    # obj_path = "debug_data/building.glb"
    # obj_path = "debug_data/dragon_head.glb"
    # obj_path = "debug_data/sample_151.ply"
    obj_path = "debug_data/sample_283.ply"
    # obj_path = "debug_data/sample_266.ply"
    # obj_path = "debug_data/nike2.glb"
    # obj_path = "debug_data/robot.glb"
    # obj_path = "debug_data/sample_143.ply"
    # obj_path = "debug_data/muscle_chicken.glb"
    # obj_path = "debug_data/muscle_man.glb"
    # obj_path = "debug_data/gothic_tower.glb"
    # obj_path = "debug_data/sample_227.ply"
    # obj_path = "debug_data/_tmglb.glb"
    # obj_path = "debug_data/sample_128.ply"
    # obj_path = "debug_data/skull.glb"
    # obj_path = "debug_data/skull2.glb"
    # obj_path = "debug_data/skull_correct.glb"
    # obj_path = "debug_data/skull.ply"
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
        np.save(os.path.join('latent.npy'), latent.detach().cpu().numpy())
        
        # check rendering
        # render_mode = 'stf' # you can change this to 'nerf'
        render_debug = True
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
            # print(f"===DONE {obj_id}===")
            exit()
        # exit()