import os
import glob
import sys
import pyvips

def cvt_jpg2tiff(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    src_files = glob.glob(os.path.join(src_dir, '*.jpg'))
    for src_file in src_files:
        dst_file = os.path.join(dst_dir, os.path.basename(src_file).replace('.jpg', '.tiff'))

        # convert jpg to tiff
        filename = os.path.splitext(os.path.basename(src_file))[0]
        print(filename)
        im = pyvips.Image.new_from_file(src_file)
        im.write_to_file(dst_file, pyramid=True, tile=True, bigtiff=True, compression="none")
    print('Done.')

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 3:
        print('Usage: python cvt_jpg2tiff.py src_dir dst_dir')
        sys.exit(0)
    src_dir = args[1]
    dst_dir = args[2]
    cvt_jpg2tiff(src_dir, dst_dir)
        
