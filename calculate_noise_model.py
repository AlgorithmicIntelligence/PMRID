import os

import matplotlib.pyplot as plt
import numpy as np
import argparse
from numba import jit
import numba as nb


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")

    # Add arguments
    parser.add_argument("-f", "--folder_path", help="Description of foo argument")
    parser.add_argument("-b", "--bar", type=int, help="Description of bar argument")

    # Parse arguments
    args = parser.parse_args()

    return args


def main(args):
    walk_all_folders_and_calculate(args.folder_path)


def parsing_files(txt_path=None, folder_path=None, type='sid'):
    iso_table = dict()
    ## type: 'vicore', 'sid'
    if type == 'sid':
        folder_path = os.path.dirname(txt_path)
        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            items = line.split()
            arw_file_name = items[1]
            iso = int(items[2][3:])
            
            arw_file_path = os.path.join(folder_path, arw_file_name)
            if iso in iso_table:
                iso_table[iso].append(arw_file_path)
            else:
                iso_table[iso] = [arw_file_path]
                        
    return iso_table
            
            


def walk_all_folders_and_calculate(folder):
    gain_list = list()
    k_list = list()
    sigma2_list = list()

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.raw') and '_gain' in file:
                num_files = len(os.listdir(root))
                if num_files < 30:
                    break
                gain = int(file.split('_gain')[1].split('_')[0])
                width = int(file.split('_w')[1].split('_')[0])
                height = int(file.split('_h')[1].split('_')[0])
                golden_data = get_avg_img_from_folder(root, height, width)
                raw_paths = [os.path.join(root, raw_name) for raw_name in os.listdir(root)]

                print(f'calculate folder: {root}, num of files: {num_files}, gain: {gain}')
                k, sigma2 = calculate_k_sigma(raw_paths, golden_data, height, width, gain)
                print(f'gain: {gain}, k: {k}, sigma2: {sigma2}')
                gain_list.append(gain)
                k_list.append(k)
                sigma2_list.append(sigma2)
                break

    print(f'gain = {gain_list}')
    print(f'k = {k_list}')
    print(f'sigma2 = {sigma2_list}')

    coefficients_k = find_poly_coefficients(gain_list, k_list, degree=1)
    coefficients_sigma2 = find_poly_coefficients(gain_list, sigma2_list, degree=2)
    poly_k = np.poly1d(coefficients_k)
    x_values = np.linspace(0, 1024*32, 100)
    y_k = poly_k(x_values)
    y_sigma2 = np.polyval(coefficients_sigma2, x_values)

    # poly_sigma2 = np.poly2d(coefficients_sigma2)

    # x_min = 0
    # x_max = 1024
    #
    # y_values = np.array(y_values, dtype=np.float64)
    #
    # x_values_poly = np.linspace(x_min, x_max, x_max-x_min)
    # y_values_poly = poly_function(x_values_poly)
    plt.scatter(gain_list, k_list, s=0.5, color='blue', label='Data')
    plt.plot(x_values, y_k, color='red', label='Polynomial Fit')
    plt.xlabel('gain')  # 设置X轴标签
    plt.ylabel('k')  # 设置Y轴标签
    plt.title('Noise Analysis')  # 设置标题
    plt.legend()
    plt.grid(True)  # 显示网格
    plt.xlim(0, 1024*32)
    plt.show()  # 显示图表

    plt.scatter(gain_list, sigma2_list, s=0.5, color='blue', label='Data')
    plt.plot(x_values, y_sigma2, color='red', label='Polynomial Fit')
    plt.xlabel('gain')  # 设置X轴标签
    plt.ylabel('sigma2')  # 设置Y轴标签
    plt.title('Noise Analysis')  # 设置标题
    plt.legend()
    plt.grid(True)  # 显示网格
    plt.xlim(0, 1024*32)
    plt.show()  # 显示图表

@jit(nopython=True)
def calculate_dict(square_difference, golden_data, golden_differences_map, golden_differences_cnt_map):
    for idx2, golden_value in enumerate(golden_data):
        if golden_value in golden_differences_map:
            # golden_differences_cnt_map[golden_value]['count'] += 1
            # golden_differences_map[golden_value]['sum_square_difference'] += square_difference
            golden_differences_cnt_map[golden_value] += 1
            golden_differences_map[golden_value] += square_difference[idx2]
        else:
            # golden_differences_map[golden_value] = dict()
            # golden_differences_cnt_map[golden_value]['count'] = 1
            # golden_differences_map[golden_value]['sum_square_difference'] = square_difference
            golden_differences_cnt_map[golden_value] = 1
            golden_differences_map[golden_value] = square_difference[idx2]


def calculate_k_sigma(raw_paths, golden_data, height=1080, width=1920, gain=None):
    fig_folder = 'results'
    if not os.path.isdir(fig_folder):
        os.mkdir(fig_folder)
    save_fig_path = os.path.join(fig_folder, f'vi3140_gain{gain}')
    tmp_fig_path = save_fig_path
    cnt = 2
    while os.path.isfile(tmp_fig_path):
        tmp_fig_path = save_fig_path + '_' + str(cnt)
        cnt += 1
    save_fig_path = tmp_fig_path + '.png'

    # golden_data = get_avg_img_from_folder(folder_path, height, width)
    # golden_differences_map = dict()
    # golden_differences_map = nb.typed.Dict.empty(
    #     key_type=nb.types.int64,
    #     value_type=nb.typed.Dict.empty(
    #         key_type=nb.types.int64,
    #         value_type=nb.types.float64)
    # )
    golden_differences_map = nb.typed.Dict.empty(
        key_type=nb.types.uint16,
        value_type=nb.types.float64
    )
    golden_differences_cnt_map = nb.typed.Dict.empty(
        key_type=nb.types.uint16,
        value_type=nb.types.int64
    )
    for idx, raw_path in enumerate(raw_paths):
        if not raw_path.endswith('.raw'):
            continue
        # print(f'idx: {idx}, raw_name: {os.path.basename(raw_path)}')
        # raw_path = os.path.join(folder_path, raw_name)
        raw_path = '\\\\?\\' + raw_path
        raw_data = np.fromfile(raw_path, dtype="uint16").astype("int")
        raw_data = raw_data[:height*width]
        square_difference = (golden_data - raw_data)**2
        calculate_dict(square_difference, golden_data, golden_differences_map, golden_differences_cnt_map)
        # for idx2, golden_value in enumerate(golden_data):
        #     if (idx2+1) % 500 == 0:
        #         print(f'idx2: {idx2+1}')
        #     if golden_value in golden_differences_map:
        #         golden_differences_map[golden_value]['count'] += 1
        #         golden_differences_map[golden_value]['sum_square_difference'] += square_difference
        #     else:
        #         golden_differences_map[golden_value] = dict()
        #         golden_differences_map[golden_value]['count'] = 1
        #         golden_differences_map[golden_value]['sum_square_difference'] = square_difference

    x_values = list()
    y_values = list()
    x_values_outlier = list()
    y_values_outlier = list()

    for key in np.array(list(golden_differences_map.keys()), dtype=np.uint16):
        # if key > np.iinfo(np.uint16).max or key < np.iinfo(np.uint16).min:
        #     raise Exception('overflow')
        # golden_differences_map[key] /= golden_differences_cnt_map[key]
        cnt = golden_differences_cnt_map[key]
        variance = golden_differences_map[key] / cnt
        # variance = golden_differences_map[key]
        print(f'key: {key}, variance: {variance}, cnt: {cnt}')
        if cnt <= len(raw_paths) * 300:
            x_values_outlier.append(key)
            y_values_outlier.append(variance)
    # for key, value in golden_differences_map.items():
    #     variance = value['sum_square_difference']/value['count']
        else:
            x_values.append(key)
            y_values.append(variance)

    coefficients = find_poly_coefficients(x_values, y_values, degree=1)
    poly_function = np.poly1d(coefficients)
    print(f'poly[0] = {poly_function(0)}, poly[500] = {poly_function(500)}')

    x_min = 0
    x_max = 1024

    y_values = np.array(y_values, dtype=np.float64)

    x_values_poly = np.linspace(x_min, x_max, x_max-x_min)
    y_values_poly = poly_function(x_values_poly)
    plt.scatter(x_values, y_values, s=0.5, color='blue', label='Data')
    plt.scatter(x_values_outlier, y_values_outlier, s=0.5, color='red', label='OutlierData')
    plt.plot(x_values_poly, y_values_poly, color='red', label='Polynomial Fit')
    # plt.scatter(x_values, y_values, s=1, c='red')
    plt.xlabel('E[X*]')  # 设置X轴标签
    plt.ylabel('Var(X*)')  # 设置Y轴标签
    plt.title('Noise Analysis')  # 设置标题
    plt.legend()
    plt.grid(True)  # 显示网格
    plt.xlim(x_min, x_max)
    plt.savefig(save_fig_path)
    plt.show()  # 显示图表
    plt.clf()

    return coefficients


def find_poly_coefficients(x_values, y_values, degree=1):
    # Perform linear regression to fit a polynomial
    coefficients = np.polyfit(x_values, y_values, degree)

    # Print the coefficients
    # print("Coefficients:", coefficients)

    # Generate the polynomial function
    # poly_function = np.poly1d(coefficients)

    # Print the polynomial function
    # print("Polynomial function:", poly_function)
    return coefficients




def get_avg_img_from_folder(folder_path, height=1080, width=1920):
    raw_data_cnt = 0
    avg_data = np.zeros(height*width, dtype=np.int64)
    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.raw'):
            continue
        file_path = os.path.join(folder_path, file_name)
        file_path = '\\\\?\\' + file_path
        raw_data = np.fromfile(file_path, dtype="uint16").astype("int")
        raw_data = raw_data[:height*width]
        avg_data = avg_data + raw_data
        raw_data_cnt += 1
    if raw_data_cnt == 0:
        raise Exception('The folder has no .raw files.')
    avg_data = np.round(avg_data/raw_data_cnt)
    return avg_data.astype(np.uint16)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

import numpy as np

def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

# Define kernel size
kernel_size = 5

# Build Gaussian kernel
gaussian_kernel_5x5 = gaussian_kernel(kernel_size)

# Print the kernel
print("Gaussian Kernel 5x5:")
print(gaussian_kernel_5x5)