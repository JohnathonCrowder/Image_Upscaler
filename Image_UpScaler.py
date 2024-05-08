import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QProgressBar
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PIL import Image
from super_image import EdsrModel, ImageLoader
from torchvision.transforms import ToPILImage

class ImageUpscaler(QThread):
    progress_signal = pyqtSignal(int)
    def __init__(self, input_directory):
        super().__init__()
        self.input_directory = input_directory
        self.upscaled_directory = os.path.join(input_directory, "UpScaled")
        self.model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
        self.to_pil = ToPILImage()

    def upscale_image(self, image, patch_size=256):
        # Get the width and height of the input image
        width, height = image.size

        # Create a new output image with doubled width and height
        output_image = Image.new("RGB", (width * 2, height * 2))

        # Iterate over the image in patches
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                # Extract the current patch from the input image
                patch = image.crop((x, y, x + patch_size, y + patch_size))

                # Load the patch into the model
                inputs = ImageLoader.load_image(patch)

                # Perform upscaling on the patch using the model
                preds = self.model(inputs)

                # Convert the upscaled patch tensor to a PIL image
                patch_output = self.to_pil(preds.squeeze(0))

                # Paste the upscaled patch into the output image at the corresponding position
                output_image.paste(patch_output, (x * 2, y * 2))

        # Return the upscaled output image
        return output_image

    def upscale_directory(self):
        # Create the "UpScaled" folder within the input directory
        os.makedirs(self.upscaled_directory, exist_ok=True)

        # Get a list of image files in the input directory
        image_files = [file for file in os.listdir(self.input_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        total_images = len(image_files)

        # Iterate over each image file in the input directory
        for index, image_file in enumerate(image_files, start=1):
            # Construct the input and output file paths
            input_filepath = os.path.join(self.input_directory, image_file)
            output_filepath = os.path.join(self.upscaled_directory, f"{index}.png")

            # Open the input image using PIL
            image = Image.open(input_filepath)

            # Perform upscaling on the input image using the upscale_image method
            output_image = self.upscale_image(image)

            # Save the upscaled image to the output file path
            output_image.save(output_filepath)

            print(f"Done Upscaling Image: {index}/{total_images}")

        print(f"Upscaled {total_images} images successfully.")

    def run(self):
        os.makedirs(self.upscaled_directory, exist_ok=True)
        image_files = [file for file in os.listdir(self.input_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)

        for index, image_file in enumerate(image_files, start=1):
            input_filepath = os.path.join(self.input_directory, image_file)
            output_filepath = os.path.join(self.upscaled_directory, f"{index}.png")

            image = Image.open(input_filepath)
            output_image = self.upscale_image(image)
            output_image.save(output_filepath)

            progress = int((index / total_images) * 100)
            self.progress_signal.emit(progress)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Upscaler")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        self.label_directory = QLabel("Select a directory to upscale images:")
        layout.addWidget(self.label_directory)

        self.label_image_count = QLabel("Images found: 0")
        layout.addWidget(self.label_image_count)

        self.label_status = QLabel("")
        layout.addWidget(self.label_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()

        self.button_select = QPushButton("Select Directory")
        self.button_select.clicked.connect(self.select_directory)
        button_layout.addWidget(self.button_select)

        self.button_upscale = QPushButton("Upscale Images")
        self.button_upscale.clicked.connect(self.upscale_images)
        self.button_upscale.setEnabled(False)
        button_layout.addWidget(self.button_upscale)

        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.upscaler = None
        self.hide_progress_timer = QTimer()
        self.hide_progress_timer.setInterval(5000)  # 5000 milliseconds = 5 seconds
        self.hide_progress_timer.timeout.connect(self.hide_progress_bar)

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            image_files = [file for file in os.listdir(directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_count = len(image_files)
            self.label_directory.setText(f"Selected Directory: {directory}")
            self.label_image_count.setText(f"Images found: {image_count}")
            self.button_upscale.setEnabled(True)
            self.upscaler = ImageUpscaler(directory)
            self.upscaler.progress_signal.connect(self.update_progress)

    def upscale_images(self):
        if self.upscaler:
            self.label_status.setText("Upscaling images...")
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.upscaler.start()
        else:
            self.label_status.setText("No directory selected.")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if value == 100:
            self.label_status.setText("Upscaling completed.")
            self.hide_progress_timer.start()

    def hide_progress_bar(self):
        self.progress_bar.setVisible(False)
        self.hide_progress_timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())