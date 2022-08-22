# Code for data preprocessing

## Environment

```bash
pip install numpy
pip install opencv-python
```

## Usage

### `cut_tumor_regions.py`: extract annotated tumor regions of all WSIs.

- get the usage information of scripts

```bash
python cut_tumor_regions.py --help
```

- fill the irrelevant areas with white color

```bash
python cut_tumor_regions.py --wsi_dir_path ./WSIs --output_dir_path ./tumour-regions
```

- not fill the irrelevant areas

```bash
python cut_tumor_regions.py --wsi_dir_path ./WSIs --output_dir_path ./tumour-regions --not_filled_other_regions
```

### `cut_patches.py`: cut patches with fixed size from all extracted annotated tumor regions.

- get the usage information of scripts

```bash
python cut_patches.py --help
```

- cut patches with size of ![](https://render.githubusercontent.com/render/math?math=256\times256) and the patches with blank ratio greater than 0.3 will be discarded, you can modify the parameters for your research

```bash
python cut_patches.py --tumour_region_dir_path ./tumour-regions --size 256 --max_blank_ratio 0.3 --output_dir_path ./patches
```
