import os


def rename_files(source_folder):
    """
        Renames files correctly only if file contains in the end _2345.png
        Otherwise result is unpredicted
    """

    for file_name_old in os.listdir(source_folder):
        if os.path.isfile(os.path.join(source_folder, file_name_old)):
            file_name_without_ext, file_extension = os.path.splitext(file_name_old)
            parts = file_name_without_ext.split("_")
            if len(parts) >= 2:
                file_path_new = os.path.join(source_folder, parts[len(parts) - 1] + file_extension)
                file_path_old = os.path.join(source_folder, file_name_old)
                os.rename(file_path_old, file_path_new)


# Example of usage
# rename_files("data/test_mask")

