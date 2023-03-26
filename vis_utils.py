import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import cv2
from scipy.ndimage import zoom
import torch

def create_label_colormap():
    """
    return:
        colormap: random color map
    """
    return np.asarray([
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],])

def label_to_color_image(label):
    # print(label.shape)
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

def vis_segmentation(image, seg_map, gt, save_path, legend=False):
    """
    输入图片和分割 mask 的可视化.
    """
    LABEL_NAMES = np.asarray(['background','Aorta' , 'Gallbladder', 'Kidney(L)', 'Kidney(R)', 'Liver', 'Pancreas', 'Spleen', 'Stomach'])
    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    gt = label_to_color_image(gt).astype(np.uint8)
    seg_image = label_to_color_image(seg_map).astype(np.uint8)

    plt.figure(figsize=(4, 10))
    if legend:
        grid_spec = gridspec.GridSpec(4, 1, height_ratios=[6, 6, 6, 6])
    else:
        grid_spec = gridspec.GridSpec(3, 1, height_ratios=[6, 6, 6])


    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    # plt.title('input image')

    plt.subplot(grid_spec[1])
    plt.imshow(image)
    plt.imshow(gt, alpha=0.7)
    plt.axis('off')
    # plt.title('ground truth overlay')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    # plt.title('segmentation overlay')

    if legend:
        unique_labels = np.unique(seg_map)
        ax = plt.subplot(grid_spec[3])
        plt.imshow(FULL_COLOR_MAP.astype(np.uint8), interpolation='nearest')
        ax.yaxis.tick_right()
        plt.yticks(range(len(FULL_COLOR_MAP)), LABEL_NAMES)
        plt.xticks([], [])
        ax.tick_params(width=0.0)
        plt.grid('off')
        # plt.show()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return


def vis_segmentation2(image, seg_map, save_path):
    """
    输入图片和分割 mask 的统一可视化.
    """
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.figure()
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    # plt.show()
    plt.savefig(save_path)
    return

def vis_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        # print(f'num img {image.shape[0]}')
        for ind in range(image.shape[0]):
            if ind % 10 != 0:
                continue
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred

                # img = slice
                img = image[ind, :, :]
                # img = (slice - slice.min()) / (slice.max() - slice.min())
                gt = label[ind, ...]
                # print(img.shape, pred.shape, gt.shape)
                # print(gt)
                # img = img.cpu().detach().numpy()

                assert test_save_path
                save_path = test_save_path + '/' + case + f"_slice{ind}.png"

                vis_segmentation(img, pred, gt, save_path)
                # vis_segmentation2(img, gt)

    else:
        raise NotImplementedError
        # input = torch.from_numpy(image).unsqueeze(
        #     0).unsqueeze(0).float().cuda()
        # net.eval()
        # with torch.no_grad():
        #     out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        #     prediction = out.cpu().detach().numpy()

    return