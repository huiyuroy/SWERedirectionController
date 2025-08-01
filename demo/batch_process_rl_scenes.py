import gc
import os

import pyrdw.generator as generator
from common import data_path


def preprocess_scenes():
    """
    Preprocess all rl scenes.


    Returns:

    """
    paths = [data_path + '\\rl\\vir', data_path + '\\rl\\phy']
    for f_path in paths:
        f_files = [os.path.join(f_path, file) for file in os.listdir(f_path) if 'json' in file]

        for f_file in f_files:
            dis = True
            vis = True
            vis_grid = True if 'phy' in f_file else False

            if 'empty20000x20000' in f_file:
                vis_grid = False
                dis = False
                vis = False
            scene = generator.load_scene_contour(f_file)
            scene.update_segmentation()
            scene.update_grids_precompute_attr(enable_vis=vis,
                                                enable_vis_grid=vis_grid,
                                                enable_discrete=dis)
            generator.save_scene(scene, f_path)
            gc.collect()


if __name__ == '__main__':
    preprocess_scenes()

    print("--process done!--")
