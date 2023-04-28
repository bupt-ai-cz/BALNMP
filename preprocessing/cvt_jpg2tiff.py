"""
1. install convert:
    https://github.com/ImageMagick/ImageMagick

2. install tiff-tools (ubuntu 18.04):
    pip install tifftools

If you have size limit error, try to increase the limit in: /etc/ImageMagick-6/policy.xml
check: https://github.com/ImageMagick/ImageMagick/issues/396#issuecomment-319569255

"""
import os
import glob
import sys

def cvt_jpg2tiff(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    src_files = glob.glob(os.path.join(src_dir, '*.jpg'))
    for src_file in src_files:
        dst_file = os.path.join(dst_dir, os.path.basename(src_file).replace('.jpg', '.tiff'))

        # convert jpg to tiff

        cmd = f'convert {src_file} -define tiff64:tile-geometry=256x256 -compress none "ptif:{dst_file}"'
        print(cmd)
        os.system(cmd)

        # add metadata to tiff
        cmd = 'tifftools set -y -s ImageDescription "Iscan Coreo |AppMag = 20" ' + dst_file
        print(cmd)
        os.system(cmd)
    print('Done.')

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 3:
        print('Usage: python cvt_jpg2tiff.py src_dir dst_dir')
        sys.exit(0)
    src_dir = args[1]
    dst_dir = args[2]
    cvt_jpg2tiff(src_dir, dst_dir)
        