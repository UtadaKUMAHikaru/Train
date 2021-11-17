cd data/
bash download_cyclegan_dataset.sh monet2photo
cd ../implementations/cyclegan/
python3 cyclegan.py --dataset_name monet2photo