import os

PIPE = "│"
ELBOW = "└──"
TEE = "├──"
PIPE_PREFIX = "│   "
SPACE_PREFIX = "    "


def print_directory_tree(root_path, padding=''):
    """Recursively prints a directory tree starting at root_path"""
    # Get all the files and directories at this path
    files_and_directories = os.listdir(root_path)

    # Create a list to hold the lines of the directory tree
    lines = []

    # Print each file or directory here, indented by the current padding
    for i, file_or_directory in enumerate(files_and_directories):
        if i == len(files_and_directories) - 1:
            # This is the last item in the list, so we should print an elbow
            prefix = ELBOW
            new_padding = padding + SPACE_PREFIX
        else:
            # This is not the last item, so we should print a tee
            prefix = TEE
            new_padding = padding + PIPE_PREFIX

        # Print the item with the appropriate prefix and padding
        full_path = os.path.join(root_path, file_or_directory)
        if os.path.isfile(full_path):
            lines.append(padding + prefix + file_or_directory + "\n")
        else:
            lines.append(padding + prefix + file_or_directory + "/\n")
            # If this item is a directory, recursively print its contents
            lines.extend(print_directory_tree(full_path, new_padding))

    # Return the lines of the directory tree
    return lines


# Example usage: print the tree starting at the current directory
directory_tree = print_directory_tree('./src/')

# Write the directory tree to a text file
with open('directory_structure_tree.txt', 'w') as f:
    f.writelines(directory_tree)
