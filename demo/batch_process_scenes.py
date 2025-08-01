import gc
import os

import pyrdw.generator as generator
from common import data_path


def preprocess_scenes():
    """
    Preprocess all scenes.


    Returns:

    """
    p_path = data_path + "\\phy"
    p_files = [os.path.join(p_path, file) for file in os.listdir(p_path) if 'json' in file]
    p_masks = [True] * len(p_files)
    p_f_info = dict(zip(p_files, p_masks))

    v_path = data_path + "\\vir"
    v_files = [os.path.join(v_path, file) for file in os.listdir(v_path) if 'json' in file]
    f_masks = [False] * len(v_files)
    v_f_info = dict(zip(v_files, f_masks))

    all_info = p_f_info | v_f_info

    for f_file, f_mask in all_info.items():
        scene = generator.load_scene_contour(f_file)
        scene.update_segmentation()
        scene.update_grids_precompute_attr(enable_vis=True,
                                            enable_vis_grid=f_mask,
                                            enable_discrete=True)

        generator.save_scene(scene, p_path if f_mask else v_path)
        gc.collect()

if __name__ == '__main__':
    preprocess_scenes()

    print("--process done!--")