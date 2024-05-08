import os
from super_image import EdsrModel, ImageLoader
from PIL import Image
from torchvision.transforms import ToPILImage

def upscale_image(model, image, patch_size=256):
    # Get the width and height of the input image
    width, height = image.size

    # Create a new output image with doubled width and height
    output_image = Image.new("RGB", (width * 2, height * 2))

    # Create an instance of ToPILImage for converting tensors to PIL images
    to_pil = ToPILImage()

    # Iterate over the image in patches
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # Extract the current patch from the input image
            patch = image.crop((x, y, x + patch_size, y + patch_size))

            # Load the patch into the model
            inputs = ImageLoader.load_image(patch)

            # Perform upscaling on the patch using the model
            preds = model(inputs)

            # Convert the upscaled patch tensor to a PIL image
            patch_output = to_pil(preds.squeeze(0))

            # Paste the upscaled patch into the output image at the corresponding position
            output_image.paste(patch_output, (x * 2, y * 2))

    # Return the upscaled output image
    return output_image

def upscale_directory(input_directory):
    # Create the "UpScaled" folder within the input directory
    upscaled_directory = os.path.join(input_directory, "UpScaled")
    os.makedirs(upscaled_directory, exist_ok=True)

    # Get a list of image files in the input directory
    image_files = [file for file in os.listdir(input_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Load the pre-trained EDSR model for upscaling
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)

    # Iterate over each image file in the input directory
    for index, image_file in enumerate(image_files, start=1):
        # Construct the input and output file paths
        input_filepath = os.path.join(input_directory, image_file)
        output_filepath = os.path.join(upscaled_directory, f"{index}.png")

        # Open the input image using PIL
        image = Image.open(input_filepath)

        # Perform upscaling on the input image using the upscale_image function
        output_image = upscale_image(model, image)

        # Save the upscaled image to the output file path
        output_image.save(output_filepath)

        print("One image done")

    print(f"Upscaled {len(image_files)} images successfully.")

# Example usage
input_directory = r"C:\Users\Admin\Pictures\Upscal"

# Call the upscale_directory function with the input directory
upscale_directory(input_directory)