import flet as ft
from sta.io import import_image_sequence, trim_image
from sta.analysis import compute_structure_tesnsor, compute_orientation
import numpy as np

class App(ft.Column):
    def __init__(self):
        super().__init__()
        self.width = 1200
        self.controls = [
            ft.Text("Step 1: Import Image Sequences.", theme_style=ft.TextThemeStyle.HEADLINE_MEDIUM),
            ft.Row(
                [
                 ft.TextField(hint_text="Path for image sequence (..\image_)", on_submit=self._set_template, width=600),
                 ft.TextField(hint_text="Digit", on_submit=self._set_digit, expand=True, width=50),
                 ft.TextField(hint_text="Format", on_submit=self._set_format, expand=True, width=50)],
                 alignment=ft.MainAxisAlignment.CENTER),
            ft.Row(
                [
                 ft.TextField(hint_text="Start index", on_submit=self._set_start_index, expand=True, width=50),
                 ft.TextField(hint_text="End index", on_submit=self._set_end_index, expand=True, width=50),
                 ft.TextField(hint_text="Start Pixel x", on_submit=self._set_start_pixel_x, expand=True, width=20),
                 ft.TextField(hint_text="Start Pixel y", on_submit=self._set_start_pixel_y, expand=True, width=20),
                 ft.TextField(hint_text="End Pixel x", on_submit=self._set_end_pixel_x, expand=True, width=20),
                 ft.TextField(hint_text="End Pixel y", on_submit=self._set_end_pixel_y, expand=True, width=20)],
                 alignment=ft.MainAxisAlignment.CENTER),
            ft.Row(
            [
            ft.TextButton("Import image sequence", on_click=self._import_images, width=600),
            ft.TextButton("Save volume as npy", on_click=self._save_volume, width=600)]),
            ft.Text("Step 2: Compute orientations.", theme_style=ft.TextThemeStyle.HEADLINE_MEDIUM),
            ft.Row([
                ft.TextField(hint_text="Noise scale", on_submit=self._set_noise_scale, width=300),
                ft.TextButton("Compute orientations", on_click=self._compute_orientations, expand=True, width=300),
                ft.TextButton("Export data", on_click=self._export_data, expand=True, width=300)
            ]),
            ft.Text("Step 3: Estimate Compressive strength.", theme_style=ft.TextThemeStyle.HEADLINE_MEDIUM)]
        
    def _set_noise_scale(self, e):
        try:
            self.noise_scale = float(e.data)
        except ValueError:
            raise ValueError(f"Invalid noise scale: {e.data}. Must be a float.")
        if self.noise_scale < 0:
            raise ValueError(f"Invalid noise scale: {self.noise_scale}. Must be a non-negative float.")
        self.noise_scale = float(e.data)
        print("ok")
        pass
    def _compute_orientations(self, e):
        if not hasattr(self, 'volume'):
            raise ValueError("Volume must be imported before computing orientations.")
        if not hasattr(self, 'noise_scale'):
            raise ValueError("Noise scale must be set before computing orientations.")
        print("Computing structure tensor...")
        tensor = compute_structure_tesnsor(self.volume, self.noise_scale)
        print("Done")
        print("Computing orientations...")
        self.theta, self.phi = compute_orientation(tensor)
        reference_vector = [1, 0, 0]
        self.varphi = compute_orientation(tensor, reference_vector)
        print("Done")
        pass
    def _export_data(self, e):
        pass

    def _set_template(self, e):
        self.template = e.data
        print("ok")
        pass
    def _set_digit(self, e):
        try:
            self.digit = int(e.data)
        except ValueError:
            raise ValueError(f"Invalid digit: {e.data}. Must be an integer.")
        if self.digit < 0:
            raise ValueError(f"Invalid digit: {self.digit}. Must be a non-negative integer.")
        if self.digit > 9:
            raise ValueError(f"Invalid digit: {self.digit}. Must be a single digit (0-9).")
        self.digit = int(e.data)
        print("ok")

        pass
    def _set_format(self, e):
        format = e.data
        supported_formats = ["png", "jpg", "jpeg", "tiff", "bmp", "tif", "dcm"]
        if format not in supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported formats are: {supported_formats}")
        self.format = format
        print("ok")
        pass

    def _set_start_index(self, e):
        try:
            self.start_index = int(e.data)
        except ValueError:
            raise ValueError(f"Invalid start index: {e.data}. Must be an integer.")
        if self.start_index < 0:
            raise ValueError(f"Invalid start index: {self.start_index}. Must be a non-negative integer.")
        self.start_index = int(e.data)
        print("ok")
        pass
    def _set_end_index(self, e):
        try:
            self.end_index = int(e.data)
        except ValueError:
            raise ValueError(f"Invalid end index: {e.data}. Must be an integer.")
        if self.end_index < 0:
            raise ValueError(f"Invalid end index: {self.end_index}. Must be a non-negative integer.")
        self.end_index = int(e.data)
        print("ok")
        pass
    def _set_start_pixel_x(self, e):
        try:
            self.start_pixel_x = int(e.data)
        except ValueError:
            raise ValueError(f"Invalid start pixel x: {e.data}. Must be an integer.")
        if self.start_pixel_x < 0:
            raise ValueError(f"Invalid start pixel x: {self.start_pixel_x}. Must be a non-negative integer.")
        self.start_pixel_x = int(e.data)
        print("ok")
        pass
    def _set_start_pixel_y(self, e):
        try:
            self.start_pixel_y = int(e.data)
        except ValueError:
            raise ValueError(f"Invalid start pixel y: {e.data}. Must be an integer.")
        if self.start_pixel_y < 0:
            raise ValueError(f"Invalid start pixel y: {self.start_pixel_y}. Must be a non-negative integer.")
        self.start_pixel_y = int(e.data)
        print("ok")
        pass
    def _set_end_pixel_x(self, e):
        try:
            self.end_pixel_x = int(e.data)
        except ValueError:
            raise ValueError(f"Invalid end pixel x: {e.data}. Must be an integer.")
        if self.end_pixel_x < 0:
            raise ValueError(f"Invalid end pixel x: {self.end_pixel_x}. Must be a non-negative integer.")
        self.end_pixel_x = int(e.data)
        print("ok")
        pass
    def _set_end_pixel_y(self, e):
        try:
            self.end_pixel_y = int(e.data)
        except ValueError:
            raise ValueError(f"Invalid end pixel y: {e.data}. Must be an integer.")
        if self.end_pixel_y < 0:
            raise ValueError(f"Invalid end pixel y: {self.end_pixel_y}. Must be a non-negative integer.")
        self.end_pixel_y = int(e.data)
        print("ok")
        pass

    def _import_images(self, e):
        if not hasattr(self, 'template') or not hasattr(self, 'digit') or not hasattr(self, 'format'):
            raise ValueError("Template, digit, and format must be set before importing images.")
        if not hasattr(self, 'start_index') or not hasattr(self, 'end_index'):
            raise ValueError("Start and end index must be set before importing images.")
        if not hasattr(self, 'start_pixel_x') or not hasattr(self, 'start_pixel_y') or not hasattr(self, 'end_pixel_x') or not hasattr(self, 'end_pixel_y'):
            raise ValueError("Start and end pixel coordinates must be set before importing images.")
        trim = lambda x: trim_image(x, (self.start_pixel_x, self.start_pixel_y), (self.end_pixel_x, self.end_pixel_y))
        self.volume = import_image_sequence(self.template, self.start_index, self.end_index, self.digit, self.format, processing=trim)
        pass
    def _save_volume(self, e):
        if not hasattr(self, 'volume'):
            raise ValueError("Volume must be imported before saving.")
        save_path = "volume.npy"
        np.save(save_path, self.volume)
        pass

def main(page: ft.Page):
    page.title = "STA: Structure Tensor Analysis for composite structures"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.ADAPTIVE

    page.add(App())

ft.app(main)