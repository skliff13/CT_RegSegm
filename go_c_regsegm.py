from ct_reg_segmentor import CtRegSegmentor


if __name__ == '__main__':
    rs = CtRegSegmentor()

    rs.process_file('test_data/test_image.nii.gz')

    rs.process_dir('test_data/dir_with_images')
