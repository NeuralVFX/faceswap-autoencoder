import random
import numpy as np
import cv2
import glob

from torch.utils.data import *
from torch.utils.data.sampler import *
from torchvision import transforms
from skimage.transform._geometric import _umeyama as umeyama

cv2.setNumThreads(0)


############################################################################
#  Loader Utilities
############################################################################


class NormDenorm:
    # Store mean and std for transforms, apply normalization and de-normalization
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def norm(self, img, tensor=False):
        # normalize image to feed to network
        if tensor:
            return (img - float(self.mean[0])) / float(self.std[0])
        else:
            return (img - self.mean) / self.std

    def denorm(self, img, cpu=True, variable=True):
        # reverse normalization for viewing
        if cpu:
            img = img.cpu()
        if variable:
            img = img.data
        img = img.numpy().transpose(1, 2, 0)
        return img * self.std + self.mean


def cv2_open(fn):
    # Get image with cv2 and convert from bgr to rgb
    try:
        im = cv2.imread(str(fn), cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR).astype(
            np.float32) / 255
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print('Image Open Failure:' + str(fn) + '  Error:' + str(e))


def make_img_square(input_img):
    # Take rectangular image and crop to square
    height = input_img.shape[0]
    width = input_img.shape[1]
    if height > width:
        input_img = input_img[height // 2 - (width // 2):height // 2 + (width // 2), :, :]
    if width > height:
        input_img = input_img[:, width // 2 - (height // 2):width // 2 + (height // 2), :]
    return input_img


############################################################################
# Image Augmentation/Distortion Stuff
############################################################################


class FlipCV(object):
    # resize image and bbox
    def __init__(self, p_x=.5, p_y=.5):
        self.p_x = p_x
        self.p_y = p_y

    def __call__(self, sample):

        flip_x = self.p_x > random.random()
        flip_y = self.p_y > random.random()
        if not flip_x and not flip_y:
            return sample
        else:
            image = sample['image']
            if flip_x and not flip_y:
                image = cv2.flip(image, 1)
            if flip_y and not flip_x:
                image = cv2.flip(image, 0)
            if flip_x and flip_y:
                image = cv2.flip(image, -1)
            return {'image': image}


class ResizeCV(object):
    # resize image and bbox
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        image = make_img_square(image)
        image = cv2.resize(image,
                           (self.output_size, self.output_size),
                           interpolation=cv2.INTER_CUBIC)
        return {'image': image}


class TransformCV(object):
    # apply random transform to image

    def __init__(self, rot=10, height=.05, width=.05, zoom=.1):
        # store range of possible transformations
        self.rot = rot
        self.height = height
        self.width = width
        self.zoom = zoom

    def get_random_transform(self, image):
        # create random transformation matrix
        rows, cols, ch = image.shape
        height = ((random.random() - .5) * 2) * self.width
        width = ((random.random() - .5) * 2) * self.height
        rot = ((random.random() - .5) * 2) * self.rot
        zoom = (random.random() * self.zoom) + 1

        rotation_matrix = cv2.getRotationMatrix2D((cols / 2,
                                                   rows / 2),
                                                  rot,
                                                  zoom)

        rotation_matrix = np.array([rotation_matrix[0],
                                    rotation_matrix[1],
                                    [0, 0, 1]])
        tx = width * cols
        ty = height * rows

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        transform_matrix = np.dot(translation_matrix, rotation_matrix)
        return transform_matrix

    def __call__(self, sample):
        # get transform and apply
        image_a = sample['image']
        rows, cols, ch = image_a.shape
        transform_matrix = self.get_random_transform(image_a)

        image_a = cv2.warpAffine(image_a,
                                 transform_matrix[:2, :],
                                 (cols, rows),
                                 borderMode=cv2.BORDER_REPLICATE,
                                 flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)

        return {'image': image_a}


class DistortCV(object):
    # distort image, return both distorted and non-distorted
    def __init__(self, res=64, crop_ratio=0):
        self.res = res
        self.crop_ratio = crop_ratio

    def __call__(self, sample):
        image = sample['image']
        res_scale = 256 // 64
        interp_param = 80 * res_scale
        interp_slice = slice(interp_param // 10, 9 * interp_param // 10)
        dst_pnts_slice = slice(0, 65 * res_scale, 16 * res_scale)

        # apply crop ratio
        coverage = ((res_scale * 27) - ((res_scale * 27) * self.crop_ratio))
        rand_coverage = np.random.randint(20) + coverage

        # make distortion grid
        ramp = np.linspace(128 - rand_coverage, 128 + rand_coverage, 5)
        gridx = np.broadcast_to(ramp, (5, 5))
        gridy = gridx.T
        rand_scale = np.random.uniform(5., 6.2)
        gridx = gridx + np.random.normal(size=(5, 5), scale=rand_scale)
        gridy = gridy + np.random.normal(size=(5, 5), scale=rand_scale)
        interp_gridx = cv2.resize(gridx,
                                  (interp_param,
                                   interp_param))[interp_slice, interp_slice].astype('float32')
        interp_gridy = cv2.resize(gridy,
                                  (interp_param,
                                   interp_param))[interp_slice, interp_slice].astype('float32')

        # apply distortion
        warped_image = cv2.remap(image, interp_gridx, interp_gridy, cv2.INTER_LINEAR)

        # generate matrix with umeyama
        src_coords = np.stack([gridx.ravel(), gridy.ravel()], axis=-1)
        dst_coords = np.mgrid[dst_pnts_slice, dst_pnts_slice].T.reshape(-1, 2)
        mat = umeyama(src_coords, dst_coords, True)[0:2]

        # apply matrix
        target_image = cv2.warpAffine(image, mat, (256, 256))

        warped_image = cv2.resize(warped_image,
                                  (self.res, self.res),
                                  interpolation=cv2.INTER_CUBIC)
        target_image = cv2.resize(target_image,
                                  (self.res, self.res),
                                  interpolation=cv2.INTER_CUBIC)

        return {'warped_image': warped_image, 'target_image': target_image}


class RandomColorCV(object):
    # distort image, return both distorted and non-distorted
    def __init__(self, all_images=[], crop_ratio=.2):
        self.all_images = all_images
        self.crop_ratio = crop_ratio
        self.mean_stats = []
        self.std_stats = []
        self.xyz_mean_stats = []
        self.xyz_std_stats = []

        print('Gathering Statistics')
        for num in range(len(all_images)):
            rand_file = self.all_images[num]
            tar_img = cv2_open(rand_file)

            cr = int(256 * crop_ratio)
            tar_img = cv2.resize(tar_img, (256, 256))

            # store mean and std
            self.mean_stats.append(np.mean(tar_img[cr:-cr, cr:-cr, :], axis=(0, 1)))
            self.std_stats.append(np.std(tar_img[cr:-cr, cr:-cr, :], axis=(0, 1)))

            # convert to XYZ color space
            tar_img = cv2.cvtColor(tar_img, cv2.COLOR_RGB2XYZ)
            # store XYZ mean and std
            self.xyz_mean_stats.append(np.mean(tar_img[cr:-cr, cr:-cr, :], axis=(0, 1)))
            self.xyz_std_stats.append(np.std(tar_img[cr:-cr, cr:-cr, :], axis=(0, 1)))
        print('Finished Gathering Statistics')

    def __call__(self, sample):
        image = sample['image']
        rand_idx = np.random.randint(len(self.all_images))

        cr = int(256 * self.crop_ratio)

        # maybe use XYZ colorspace
        use_xyz = np.random.choice([True, False])
        if use_xyz:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
            mt = self.xyz_mean_stats[rand_idx]
            st = self.xyz_std_stats[rand_idx]
        else:
            mt = self.mean_stats[rand_idx]
            st = self.std_stats[rand_idx]
        ms = np.mean(image[cr:-cr, cr:-cr, :], axis=(0, 1))
        ss = np.std(image[cr:-cr, cr:-cr, :], axis=(0, 1))

        # randomly interpolate the statistics
        ratio = np.random.uniform()
        mt = ratio * mt + (1 - ratio) * ms
        st = ratio * st + (1 - ratio) * ss

        # Apply color transfer from src to tar domain
        if ss.any() <= 1e-7 or st.any() <= 1e-7 or np.isnan(ss).any() or np.isnan(st).any():
            return {'image': image}

        result = st * (image.astype(np.float32) - ms) / (ss + 1e-7) + mt
        # push out of negative
        if result.min() < 0:
            result = result - result.min()
        # scale down below 1.0
        if result.max() > 1.0:
            result = (1.0 / result.max() * result).astype(np.float32)

        # convert_back
        if use_xyz:
            result = cv2.cvtColor(result, cv2.COLOR_XYZ2RGB)
        return {'image': result}


############################################################################
#  Dataset and Loader
############################################################################


class FaceDataset(Dataset):
    # Load Images from GeoPose3k Dataset and Apply Augmentation
    def __init__(self, path_a, path_b, transform, output_res=256):
        self.transform = transform
        self.path_list_rgb = glob.glob(str(path_a) + '/rgb/*.jpg')
        self.path_list_rgb_b = glob.glob(str(path_b) + '/rgb/*.jpg')
        self.output_res = output_res
        self.rand_color = RandomColorCV(all_images=self.path_list_rgb + self.path_list_rgb_b,
                                        crop_ratio=.2)
        self.data_transforms = transforms.Compose([ResizeCV(256),
                                                   FlipCV(p_x=.5, p_y=0),
                                                   TransformCV(rot=10, height=.05, width=.05, zoom=.1),
                                                   DistortCV(output_res)])

    def transform_set(self, image_rgb):
        # Apply augmentation
        col_dict = {'image': image_rgb}
        col_dict = self.rand_color(col_dict)
        trans_dict = {'image': col_dict['image']}
        trans_dict = self.data_transforms(trans_dict)
        warped = np.rollaxis(self.transform.norm(trans_dict['warped_image'][:, :, :3]), 2)
        target = np.rollaxis(self.transform.norm(trans_dict['target_image'][:, :, :3]), 2)
        return warped, target

    def __getitem__(self, index):
        # lookup id from permuted list, apply transform, return tensor
        image_path_rgb = self.path_list_rgb[index]
        image_rgb = cv2_open(image_path_rgb)
        warped, target = self.transform_set(image_rgb)
        warped_tensor = torch.FloatTensor(warped)
        target_tensor = torch.FloatTensor(target)
        return warped_tensor, target_tensor

    def __len__(self):
        return len(self.path_list_rgb)
