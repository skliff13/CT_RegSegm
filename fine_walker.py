import os
import numpy as np
from time import time
from skimage import io
from skimage.color import hsv2rgb
from skimage.morphology import binary_erosion, thin
from skimage.segmentation import random_walker
import regsegm_utils as reg


class FineWalker:
    """Creates an instance of FineWalker class which can be used for correction of borders  of the segmented
    lung regions in 3D Computed Tomography (CT) images of chest.
    The correction of borders is done by shrinking the 'mask' and 'not-mask' regions and performing re-segmentation of
    the newly-formed region in between which is supposed to contain the true border of lung regions.
    The re-segmentation of border region is made using Random Walker Segmentation algorithm. Here the shrunk 'mask'
    and 'not-mask' regions are treated as seed labels for the algorithm.

    # Arguments
        erosion_r: (int) Radius in pixel units (of axial slice) of the structuring element to perform the erosions.
            Defaults to 7.
        erosion_flat: (bool) If set to True, the erosions are performed in a slice-wise manner in 2D.
            Otherwise the erosions are performed in 3D with use of 3D structuring element. Defaults to True.
        thinnings: (int) Number of 2D thinning operations applied in a slice-wise manner after the erosions.
             Defaults to 5.
        kernel_size: (int) Number of axial CT slices used to run the Random Walker algorithm. When set to 1, the Random
            Walker is performed in 2D. Warning: using large values may significantly increase the processing time.
            Defaults to 1.
        stride: (int) Z-axis stride  of the sliding window to perform the Random Walker algorithm.
            Defaults to 1.

    # Examples
    `.process_image_mask('image.nii.gz', 'mask.nii.gz', 'result.nii.gz')`
    `.process_image_mask('image.nii.gz', 'mask.nii.gz')`

    """

    def __init__(self, erosion_r=7, erosion_flat=True, thinnings=5, kernel_size=1, stride=1):
        self.erosion_r = erosion_r
        self.thinnings = thinnings
        self.erosion_flat = erosion_flat
        self.kernel_size = kernel_size
        self.stride = stride

    def process_image_mask(self, image_path, mask_path, out_path=None):
        """Reads a Nifti CT image, the existing lungs mask and performs correction of borders.

        # Arguments
            image_path: (str) Path to the input file with CT image.
            mask_path: (str) Path to the input file with the existing lungs mask.
            out_path: (str or None, optional) Path to the output file.
                If not specified, the output file path is generated automatically.

        """

        im, voxel_dimensions, affine, shape0 = reg.adv_analyze_nii_read(image_path)
        im = reg.make_uint8(im)

        msk, _, _, _ = reg.adv_analyze_nii_read(mask_path)
        msk = np.array(msk > msk.min())

        inner, outer = self._preprocess_lung_mask(msk, voxel_dimensions)

        accum_inner = self._perform_random_walk(im, inner, outer)

        if out_path is None:
            out_path = self._compose_out_path(image_path)

        print('Saving to ' + out_path)
        reg.save_as_nii(accum_inner, affine, out_path)

        # print('Here are some illustrations')
        # accum_inner[accum_inner == 0] = 2
        # for c in [1, 15, 30, 60, 100, -2]:
        #     self._compose_images_for_slice(im[..., c], msk[..., c], inner[..., c], outer[..., c], accum_inner[..., c])

    def _preprocess_lung_mask(self, msk, voxel_dimensions):
        print('Eroding')
        start_time = time()
        selem = self._make_structure_element(voxel_dimensions)
        inner = binary_erosion(msk, selem=selem)
        outer = binary_erosion(np.logical_not(msk), selem=selem)
        print('Eroding took %.3f sec' % (time() - start_time))

        print('Thinning')
        start_time = time()
        for k in range(msk.shape[2]):
            inner[..., k] = thin(inner[..., k], max_iter=self.thinnings)
            outer[..., k] = thin(outer[..., k], max_iter=self.thinnings)
        print('Thinning took %.3f sec' % (time() - start_time))

        return inner, outer

    def _perform_random_walk(self, im, inner, outer):
        labels = np.zeros(im.shape, dtype='int16')
        labels[inner] = 1
        labels[outer] = 2

        accum_inner, accum = im * 0, im * 0
        print('Random walking')
        ds = self.kernel_size // 2
        start_time = time()
        for k in range(ds, im.shape[2] - self.kernel_size + ds, self.stride):
            slice_start = k - ds
            slice_end = k - ds + self.kernel_size

            for i, side in enumerate(['left', 'right']):
                x_start = (im.shape[1] // 2 - 1) * i
                x_end = x_start + im.shape[1] // 2
                im_part = im[:, x_start:x_end, slice_start:slice_end]
                lbl_part = labels[:, x_start:x_end, slice_start:slice_end]

                if np.any(lbl_part == 1) and np.any(lbl_part == 2):
                    print('%i / %i (%s)' % (k, im.shape[2], side))

                    out_labels = random_walker(im_part, lbl_part)
                    accum_inner[:, x_start:x_end, slice_start:slice_end] += out_labels == 1
                    accum[:, x_start:x_end, slice_start:slice_end] += 1

        print('Random walk took %.3f sec' % (time() - start_time))
        accum[accum == 0] = 1
        accum_inner = accum_inner * 10 // accum
        accum_inner[accum_inner < 5] = 0
        accum_inner[accum_inner > 0] = 1
        return accum_inner

    def _make_structure_element(self, voxel_dimensions):
        if self.erosion_flat:
            rz = 0
        else:
            z2xy = voxel_dimensions[2] / voxel_dimensions[0]
            rz = round(self.erosion_r / z2xy)

        xy_values = np.linspace(-1, 1, 2 * self.erosion_r + 3)
        z_values = np.linspace(-1, 1, 2 * rz + 3)
        xx, yy, zz = np.meshgrid(xy_values, xy_values, z_values)
        selem = (xx ** 2 + yy ** 2 + zz ** 2) < 1
        selem = selem[1:-1, 1:-1, 1:-1]
        return selem

    def _compose_out_path(self, image_path):
        ext = self._detect_extension(image_path)

        base = image_path[:-len(ext)] if len(ext) > 0 else image_path
        flat_char = 'f' if self.erosion_flat else ''
        out_path = f'{base}_e{self.erosion_r}{flat_char}_t{self.thinnings}_k{self.kernel_size}_s{self.stride}{ext}'

        return out_path

    @staticmethod
    def _detect_extension(path):
        known_extensions = ['.nii.gz', '.nii', '.hdr', '.img']

        for e in known_extensions:
            if path.endswith(e):
                return e

        return os.path.splitext(path)[1]

    @staticmethod
    def _compose_images_for_slice(im, msk, inner, outer, out_labels):
        def colorize_regions(image, region1, region2, h1=0.35, h2=0., s1=0.7, s2=0.7):
            hsv = np.zeros(image.shape + (3,), dtype=float)
            hsv[..., 2] = 0.5 + 0.5 * image / 255.
            hsv[..., 1] = region1 * s1 + region2 * s2
            hsv[..., 0] = h1 * region1 + h2 * region2
            return hsv2rgb(hsv)

        original_segm = colorize_regions(im, msk, 1 - msk)
        inner_and_outer = colorize_regions(im, inner, outer)
        walker_results = colorize_regions(im, out_labels == 1, out_labels == 2)

        io.imshow_collection((original_segm, inner_and_outer, walker_results))
        io.show()


if __name__ == '__main__':
    fw = FineWalker()
    for path in ['test_data/test_image.nii.gz', 'test_data/dir_with_images/image1.nii.gz',
                 'test_data/dir_with_images/image2.nii.gz']:
        fw.process_image_mask(path, path.replace('.nii.gz', '_regsegm_py.nii.gz'))
