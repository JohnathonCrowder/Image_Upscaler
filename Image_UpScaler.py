import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QProgressBar, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PIL import Image
from super_image import EdsrModel, ImageLoader
from torchvision.transforms import ToPILImage

class ImageUpscaler(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self, input_directory, scale_factor):
        super().__init__()
        self.input_directory = input_directory
        self.upscaled_directory = os.path.join(input_directory, f"UpScaled_{scale_factor}x")
        self.scale_factor = scale_factor
        self.model = self.load_model(scale_factor)
        self.to_pil = ToPILImage()
        self.image_files = [file for file in os.listdir(input_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def load_model(self, scale_factor):
        if scale_factor == 2:
            return EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
        elif scale_factor == 4:
            return EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}")

    def upscale_image(self, image, patch_size=256):
        # Get the width and height of the input image
        width, height = image.size

        # Create a new output image with scaled width and height
        output_image = Image.new("RGB", (width * self.scale_factor, height * self.scale_factor))

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
                output_image.paste(patch_output, (x * self.scale_factor, y * self.scale_factor))

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
        self.setGeometry(100, 100, 800, 600)

        main_layout = QHBoxLayout()

        # Image preview column
        self.label_image_preview = QLabel()
        self.label_image_preview.setAlignment(Qt.AlignCenter)
        self.label_image_preview.setStyleSheet("border: 2px solid black; background-color: #F0F0F0;")
        self.label_image_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.label_image_preview)

        # Sidebar menu column
        sidebar_layout = QVBoxLayout()

        self.label_input = QLabel("Select an input:")
        sidebar_layout.addWidget(self.label_input)

        self.label_status = QLabel("    ")
        sidebar_layout.addWidget(self.label_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        sidebar_layout.addWidget(self.progress_bar)

        self.button_select_directory = QPushButton("Select Directory")
        self.button_select_directory.clicked.connect(self.select_directory)
        sidebar_layout.addWidget(self.button_select_directory)

        self.button_select_image = QPushButton("Select Image")
        self.button_select_image.clicked.connect(self.select_image)
        sidebar_layout.addWidget(self.button_select_image)

        self.scale_factor = 2  # Default scale factor
        self.scale_label = QLabel("Scale Factor:")
        sidebar_layout.addWidget(self.scale_label)

        scale_layout = QHBoxLayout()
        self.scale_2x_button = QPushButton("2x")
        self.scale_2x_button.clicked.connect(lambda: self.set_scale_factor(2))
        self.scale_2x_button.setCheckable(True)
        self.scale_2x_button.setChecked(True)
        scale_layout.addWidget(self.scale_2x_button)

        self.scale_4x_button = QPushButton("4x")
        self.scale_4x_button.clicked.connect(lambda: self.set_scale_factor(4))
        self.scale_4x_button.setCheckable(True)
        scale_layout.addWidget(self.scale_4x_button)

        sidebar_layout.addLayout(scale_layout)

        self.button_upscale = QPushButton("Upscale")
        self.button_upscale.clicked.connect(self.upscale)
        self.button_upscale.setEnabled(False)
        sidebar_layout.addWidget(self.button_upscale)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar_layout)
        sidebar_widget.setFixedWidth(200)
        main_layout.addWidget(sidebar_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.upscaler = None
        self.input_path = None
        self.is_directory = False

        self.hide_progress_timer = QTimer()
        self.hide_progress_timer.setInterval(5000)
        self.hide_progress_timer.timeout.connect(self.hide_progress_bar)

        self.hide_status_timer = QTimer()
        self.hide_status_timer.setInterval(5000)
        self.hide_status_timer.timeout.connect(self.hide_status_label)

    def set_scale_factor(self, factor):
        self.scale_factor = factor
        if factor == 2:
            self.scale_2x_button.setChecked(True)
            self.scale_4x_button.setChecked(False)
        else:
            self.scale_2x_button.setChecked(False)
            self.scale_4x_button.setChecked(True)

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.input_path = directory
            self.is_directory = True
            self.label_input.setText(f"Selected Directory: {directory}")
            self.button_upscale.setEnabled(True)
            self.clear_image_preview()

    def select_image(self):
        image_file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if image_file:
            self.input_path = image_file
            self.is_directory = False
            self.label_input.setText(f"Selected Image: {image_file}")
            self.button_upscale.setEnabled(True)
            self.update_image_preview(image_file)

    def upscale(self):
        if self.input_path:
            if self.is_directory:
                self.upscale_directory()
            else:
                self.upscale_image()

    def upscale_directory(self):
        self.label_status.setText(f"Upscaling images ({self.scale_factor}x)...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.upscaler = ImageUpscaler(self.input_path, self.scale_factor)
        self.upscaler.progress_signal.connect(self.update_progress)
        self.upscaler.start()

    def upscale_image(self):
        self.label_status.setText(f"Upscaling image ({self.scale_factor}x)...")
        upscaled_image = self.upscale_single_image(self.input_path)
        self.save_upscaled_image(upscaled_image)
        self.label_status.setText("Upscaling completed.")
        self.clear_image_preview()
        self.hide_status_timer.start()

    def upscale_single_image(self, image_path):
        model = self.load_model(self.scale_factor)
        to_pil = ToPILImage()

        image = Image.open(image_path)
        width, height = image.size
        output_image = Image.new("RGB", (width * self.scale_factor, height * self.scale_factor))

        patch_size = 256
        total_patches = ((width - 1) // patch_size + 1) * ((height - 1) // patch_size + 1)
        current_patch = 0

        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = image.crop((x, y, x + patch_size, y + patch_size))
                inputs = ImageLoader.load_image(patch)
                preds = model(inputs)
                patch_output = to_pil(preds.squeeze(0))
                output_image.paste(patch_output, (x * self.scale_factor, y * self.scale_factor))

                current_patch += 1
                progress = (current_patch / total_patches) * 100
                self.label_status.setText(f"Upscaling Image ({self.scale_factor}x)... {progress:.2f}%")

        return output_image

    def load_model(self, scale_factor):
        if scale_factor == 2:
            return EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
        elif scale_factor == 4:
            return EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}")

    def save_upscaled_image(self, upscaled_image):
        output_directory = os.path.dirname(self.input_path)
        base_name, ext = os.path.splitext(os.path.basename(self.input_path))
        output_filename = f"{base_name}_upscaled_{self.scale_factor}x{ext}"
        output_path = os.path.join(output_directory, output_filename)
        upscaled_image.save(output_path)

    def hide_status_label(self):
        self.label_status.setVisible(False)
        self.hide_status_timer.stop()

    def clear_image_preview(self):
        self.label_image_preview.clear()

    def update_image_preview(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.label_image_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_image_preview.setPixmap(scaled_pixmap)

    def update_progress(self, value):
        current_value = self.progress_bar.value()
        target_value = value

        if current_value < target_value:
            self.progress_bar.setValue(current_value + 1)
            QTimer.singleShot(200, lambda: self.update_progress(target_value))
        else:
            self.progress_bar.setValue(target_value)

        total_images = len(self.upscaler.image_files)
        progress_interval = 100 // total_images
        current_image_index = (value - 1) // progress_interval

        if current_image_index < total_images:
            current_image = self.upscaler.image_files[current_image_index]
            image_path = os.path.join(self.input_path, current_image)
            self.update_image_preview(image_path)

        if value == 100:
            self.label_status.setText("Upscaling completed.")
            self.hide_progress_timer.start()

    def hide_progress_bar(self):
        self.progress_bar.setVisible(False)
        self.label_status.setVisible(False)
        self.clear_image_preview()
        self.hide_progress_timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())