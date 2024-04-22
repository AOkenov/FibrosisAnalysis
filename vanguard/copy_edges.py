import os
import shutil


def copy_files_with_pattern(src_dir, dest_dir, pattern, new_pattern):
    """
    Copy files from source directory to destination directory
    if their names contain the given pattern, with a new name.
    
    Parameters:
        src_dir (str): Source directory path.
        dest_dir (str): Destination directory path.
        pattern (str): Pattern to search for in file names.
        new_pattern (str): New pattern to replace the old one.
    """
    # Ensure the destination directory exists, create it if not
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate over files in source directory
    for filename in os.listdir(src_dir):
        # Check if file name contains the pattern
        if pattern in filename:
            # Construct source and destination file paths
            src_file = os.path.join(src_dir, filename)
            dest_file = os.path.join(dest_dir, filename.replace(pattern,
                                                                new_pattern))
            
            os.rename(src_file, dest_file)

# Example usage:
dirs = ['E10615_MYH7/', 'E10621_ABCC9/', 'E10691_RBM20/', 'E10788_LMNA/',
        'E10884/', 'E10927_MYBPC3/', 'E11442_TTN/', 'E11443_LMNA/',
        'E11444_LMNA/', 'E11971_MYH7/']


path = '/home/arstan/Projects/Fibrosis/fibrosisanalysis/data/'
for dirname in dirs:
    source_directory = path + dirname + 'Stats/'
    destination_directory = source_directory
    pattern_to_search = "_WABL"
    pattern_to_replace = "_NABL"

    copy_files_with_pattern(source_directory, destination_directory,
                            pattern_to_search, pattern_to_replace)
