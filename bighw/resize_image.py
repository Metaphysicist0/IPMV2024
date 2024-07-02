from PIL import Image, UnidentifiedImageError
import os

def convert_and_resize_images(directory, target_size=(400, 300), output_format='PNG'):
    """
    Resize all images in the specified directory to the target size and convert them to PNG format.
    """
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            filepath = os.path.join(directory, filename)
            output_filepath = os.path.join(directory, os.path.splitext(filename)[0] + '.png')
            try:
                with Image.open(filepath) as img:
                    img = img.resize(target_size, Image.ANTIALIAS)
                    img.save(output_filepath, format=output_format)
                    print(f"Converted to PNG and resized, then saved as: {output_filepath}")
            except UnidentifiedImageError:
                print(f"Could not open {filename} as it is not a valid image file.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        print("Usage: python convert_and_resize_images.py [path_to_directory]")
        sys.exit(1)

    convert_and_resize_images(directory)
    print("All images have been converted to PNG and resized.")

