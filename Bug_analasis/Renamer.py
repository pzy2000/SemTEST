import os


def sort_and_rename_subdirs(dir_path):
    # Get list of subdirectories with their full paths
    subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    # Sort the subdirectories by last modified time
    subdirs_sorted = sorted(subdirs, key=lambda x: os.path.getmtime(x))

    # Rename the directories to add a number to the end
    for i, subdir in enumerate(subdirs_sorted, 1):
        new_name = f"{subdir}-{i}"
        os.rename(subdir, new_name)

    # Return the renamed subdirectories for verification
    return os.listdir(dir_path)


if __name__ == '__main__':
    # Replace '/path/to/your/directory' with the path to your 'mobilenet' directory
    directory_path = '/root/zmx/COMET-master/data/working_dir/COMET/results/models'
    renamed_subdirs = sort_and_rename_subdirs(directory_path)
    print(renamed_subdirs)
