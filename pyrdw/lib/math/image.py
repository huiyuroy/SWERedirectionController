from pyrdw.lib.math import *

gaussian_blur_kernel3x3 = np.array([[1 / 16, 2 / 16, 1 / 16],
                                    [2 / 16, 4 / 16, 2 / 16],
                                    [1 / 16, 2 / 16, 1 / 16]])

gaussian_blur_kernel5x5 = np.array([[2, 4, 5, 4, 2],
                                    [4, 9, 12, 9, 4],
                                    [5, 12, 15, 12, 5],
                                    [4, 9, 12, 9, 4],
                                    [2, 4, 5, 4, 2]]) / 159

sobel_x_kernel3x3 = np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]])

sobel_y_kernel3x3 = np.array([[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]])

surround_index_3x3 = ((-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1))  # col row


def blur_image(ori_img, h, w):
    iteration = 2
    grad_diff = 0.1
    kernel_size = 5
    kernel_template = gaussian_blur_kernel5x5
    half_k = int((kernel_size - 1) * 0.5)
    padding_data = np.zeros((h + half_k * 2, w + half_k * 2))
    padding_data[half_k:h + half_k, half_k:w + half_k] = ori_img.copy()
    itera_data = np.zeros((h, w))
    continue_itera = True

    while continue_itera:
        all_smooth = True
        # past_itera = itera_data.copy()
        last_l1_grad = calc_img_grad_l1(itera_data, h, w)
        for i in range(h):
            for j in range(w):
                if padding_data[i + half_k, j + half_k] == 0:
                    itera_data[i, j] = padding_data[i + half_k, j + half_k]
                else:
                    local_data = padding_data[i:i + kernel_size, j:j + kernel_size]
                    test_data = np.ones((kernel_size, kernel_size)) * padding_data[i + half_k, j + half_k]
                    if not np.sum(local_data - test_data):
                        all_smooth = False
                    itera_data[i, j] = np.sum(local_data * kernel_template)
        # if abs(np.sum(itera_data - past_itera)) < blur_diff:
        #     continue_itera = False
        l1_grad = calc_img_grad_l1(itera_data, h, w)
        if all_smooth and abs(np.sum(l1_grad - last_l1_grad) / (h * w)) < grad_diff:
            continue_itera = False

        padding_data[half_k:h + half_k, half_k:w + half_k] = itera_data

    return padding_data[half_k:h + half_k, half_k:w + half_k]


def calc_img_grad(ori_img, h, w):
    kernel_size = 3
    kernel_template_x = sobel_x_kernel3x3
    kernel_template_y = sobel_y_kernel3x3
    half_k = int((kernel_size - 1) * 0.5)
    padding_data = np.zeros((h + half_k * 2, w + half_k * 2))
    padding_data[half_k:h + half_k, half_k:w + half_k] = ori_img.copy()
    itera_data = np.zeros((h, w, 2))
    for i in range(h):
        for j in range(w):
            local_data = padding_data[i:i + kernel_size, j:j + kernel_size]
            dx = np.sum(local_data * kernel_template_x)
            dy = np.sum(local_data * kernel_template_y)
            itera_data[i, j, :] = np.array([dx, dy])
    return itera_data


def calc_img_grad_l1(ori_img, h, w):
    kernel_size = 3
    kernel_template_x = sobel_x_kernel3x3
    kernel_template_y = sobel_y_kernel3x3
    half_k = int((kernel_size - 1) * 0.5)
    padding_data = np.zeros((h + half_k * 2, w + half_k * 2))
    padding_data[half_k:h + half_k, half_k:w + half_k] = ori_img.copy()
    itera_data = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            local_data = padding_data[i:i + kernel_size, j:j + kernel_size]
            dx = np.sum(local_data * kernel_template_x)
            dy = np.sum(local_data * kernel_template_y)
            itera_data[i, j] = abs(dx) + abs(dy)
    return itera_data
