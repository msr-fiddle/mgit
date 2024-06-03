import glob
import os
import random
import shutil


def create_fl_dataset(src_dataset_dir, tgt_dataset_dir, num_workers):
    random.seed(42)
    src_train_dataset_dir = os.path.join(src_dataset_dir, 'train')
    all_class_names = []
    for rootdir, dirs, files in os.walk(src_train_dataset_dir):
        for subdir in dirs:
            all_class_names.append(subdir)
    random.shuffle(all_class_names)
    all_class_names_partitioned = [all_class_names[i::num_workers]
                                   for i in range(num_workers)]

    for partition in ['train', 'val']:
        for i in range(num_workers):
            for class_name in all_class_names:
                src_dataset_subdir = os.path.join(src_dataset_dir,
                                                  partition,
                                                  class_name)
                tgt_dataset_subdir = os.path.join(tgt_dataset_dir,
                                                  f'worker{i}',
                                                  partition,
                                                  class_name)
                # Make directory.
                os.makedirs(tgt_dataset_subdir, exist_ok=True)

                # Copy a single image if class_name not in worker's allocated classes.
                src_filenames = glob.glob(f'{src_dataset_subdir}/*')
                if class_name not in all_class_names_partitioned[i]:
                    src_filenames = src_filenames[:1]

                # Copy everything from src_dataset_subdir to tgt_dataset_subdir.
                print(f"Copying files ({src_dataset_subdir} -> {tgt_dataset_subdir})...")
                for src_filename in src_filenames:
                    shutil.copy(src_filename, tgt_dataset_subdir)


if __name__ == '__main__':
    create_fl_dataset('../data/imagenet', '../data/fl/imagenet', num_workers=40)
