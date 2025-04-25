import flet as ft
import numpy as np
import sys
import re
import pandas as pd
from sta.io import import_image_sequence, trim_image
from sta.analysis import compute_structure_tesnsor, compute_orientation, compute_static_data
from sta.simulation import MaterialParams, estimate_compression_strength_from_profile

class ConsoleOutput:
    def __init__(self, text_control):
        self.text_control = text_control

    def write(self, message):
        self.text_control.value += str(message)
        self.text_control.update()

    def flush(self):
        pass

class App(ft.Column):
    def __init__(self, file_picker_file, file_picker_save_volume):
        super().__init__()
        self.file_picker_file = file_picker_file
        self.file_picker_save_volume = file_picker_save_volume

        self.console_output = ft.Text(
            value="", selectable=True, max_lines=20,
            expand=True, overflow="scroll"
        )
        sys.stdout = ConsoleOutput(self.console_output)

        self.template_field = ft.TextField(hint_text="Path template", read_only=True, width=600)
        self.digit_field = ft.TextField(hint_text="Digit", on_submit=self._set_digit, width=200)
        self.format_field = ft.TextField(hint_text="Format", on_submit=self._set_format, width=200)

        self._init_state()
        self.controls = self._build_ui()
        self.width = 1200
        self.spacing = 20

        self.UCS = None
        self.UCstrain = None
        self.sigma = None
        self.eps = None

    def _init_state(self):
        self.material_params = None
        self.template = ""
        self.digit = 4
        self.format = "tif"
        self.start_index = 0
        self.end_index = 0
        self.start_pixel_x = 0
        self.start_pixel_y = 0
        self.end_pixel_x = 0
        self.end_pixel_y = 0
        self.noise_scale = 10
        self.volume = None
        self.theta = None
        self.phi = None
        self.varphi = None

    def _build_ui(self):
        return [
            self._section_header("Console Output"),
            ft.Container(
                content=self.console_output,
                height=200,
                width=1200,
                bgcolor="#DADADA"
            ),
            self._section_header("Step 1: Import Image Sequences."),
            self._build_image_input_row(),
            self._build_crop_input_row(),
            self._build_import_buttons(),
            self._section_header("Step 2: Compute orientations."),
            self._build_orientation_row(),
            self._section_header("Step 3: Estimate Compressive strength."),
            self._build_material_param_inputs(),
            self._build_compute_compressive_strength_button()
        ]

    def _build_material_param_inputs(self):
        self.material_inputs = {
            "longitudinal_modulus": ft.TextField(hint_text="Longitudinal Modulus (E1)", width=150),
            "transverse_modulus": ft.TextField(hint_text="Transverse Modulus (E2)", width=150),
            "poisson_ratio": ft.TextField(hint_text="Poisson Ratio (ν12)", width=150),
            "shear_modulus": ft.TextField(hint_text="Shear Modulus (G12)", width=150),
            "tau_y": ft.TextField(hint_text="Tau_y (Shear Yield Stress)", width=150),
            "K": ft.TextField(hint_text="K (Hardening Coefficient)", width=150),
            "n": ft.TextField(hint_text="n (Hardening Exponent)", width=150),
        }
        apply_button = ft.TextButton("Apply", on_click=self._apply_material_params, width=100)
        return ft.Row([
            self.material_inputs["longitudinal_modulus"],
            self.material_inputs["transverse_modulus"],
            self.material_inputs["poisson_ratio"],
            self.material_inputs["shear_modulus"],
            self.material_inputs["tau_y"],
            self.material_inputs["K"],
            self.material_inputs["n"],
            apply_button
        ], alignment=ft.MainAxisAlignment.CENTER)

    def _section_header(self, text):
        return ft.Text(text, theme_style=ft.TextThemeStyle.HEADLINE_MEDIUM)
    
    def _build_compute_compressive_strength_button(self):
        return ft.Row([
            ft.TextButton("Compute Compressive Strength", on_click=self._compute_compressive_strength, width=280),
            ft.TextButton("Export SS-curve", on_click=self._export_sscurve, width=280),
            ], alignment=ft.MainAxisAlignment.CENTER)

    def _build_image_input_row(self):
        return ft.Row([
            ft.TextButton("Select first image", on_click=self._select_file, width=160),
            self.template_field,
            self.digit_field,
            self.format_field
        ], alignment=ft.MainAxisAlignment.CENTER)

    def _build_crop_input_row(self):
        return ft.Row([
            ft.TextField(hint_text="Start index", on_submit=self._set_start_index, width=150),
            ft.TextField(hint_text="End index", on_submit=self._set_end_index, width=150),
            ft.TextField(hint_text="Start Pixel x", on_submit=self._set_start_pixel_x, width=150),
            ft.TextField(hint_text="Start Pixel y", on_submit=self._set_start_pixel_y, width=150),
            ft.TextField(hint_text="End Pixel x", on_submit=self._set_end_pixel_x, width=150),
            ft.TextField(hint_text="End Pixel y", on_submit=self._set_end_pixel_y, width=150),
        ], alignment=ft.MainAxisAlignment.CENTER)

    def _build_import_buttons(self):
        return ft.Row([
            ft.TextButton("Import image sequence", on_click=self._import_images, width=280),
            ft.TextButton("Save volume as npy", on_click=self._save_volume, width=280)
        ], alignment=ft.MainAxisAlignment.CENTER)

    def _build_orientation_row(self):
        return ft.Row([
            ft.TextField(hint_text="Noise scale", on_submit=self._set_noise_scale, width=180),
            ft.TextButton("Compute orientations", on_click=self._compute_orientations, width=280),
            ft.TextButton("Export data", on_click=self._export_histgram, width=280)
        ], alignment=ft.MainAxisAlignment.CENTER)

    def _select_file(self, e):
        def pick_result(ev: ft.FilePickerResultEvent):
            if ev.files:
                selected_path = ev.files[0].path.replace("\\", "/")
                match = re.match(r"(.*/[^/]*?)(\d+)(\.[a-zA-Z]+)?$", selected_path)
                if match:
                    self.template = match.group(1)
                    self.digit = len(match.group(2))
                    ext = match.group(3)
                    if ext:
                        self.format = ext.lstrip(".").lower()

                    self.template_field.value = self.template
                    self.digit_field.value = str(self.digit)
                    self.format_field.value = self.format

                    self.template_field.update()
                    self.digit_field.update()
                    self.format_field.update()

                    print(f"[INFO] File selected: {selected_path}")
                    print(f"[INFO] Template: {self.template}, Digit: {self.digit}, Format: {self.format}")

        self.file_picker_file.on_result = pick_result
        self.file_picker_file.pick_files(allow_multiple=False, dialog_title="Select First Image File")

    def _set_digit(self, e):
        self.digit = self._parse_int(e.data, "Digit", 0, 9)

    def _set_format(self, e):
        fmt = e.data.lower()
        supported = ["png", "jpg", "jpeg", "tiff", "bmp", "tif", "dcm"]
        if fmt not in supported:
            print(f"[ERROR] Unsupported format: {fmt}")
            raise ValueError(f"Unsupported format: {fmt}")
        self.format = fmt

    def _set_start_index(self, e): self.start_index = self._parse_int(e.data, "Start index")
    def _set_end_index(self, e): self.end_index = self._parse_int(e.data, "End index")
    def _set_start_pixel_x(self, e): self.start_pixel_x = self._parse_int(e.data, "Start Pixel X")
    def _set_start_pixel_y(self, e): self.start_pixel_y = self._parse_int(e.data, "Start Pixel Y")
    def _set_end_pixel_x(self, e): self.end_pixel_x = self._parse_int(e.data, "End Pixel X")
    def _set_end_pixel_y(self, e): self.end_pixel_y = self._parse_int(e.data, "End Pixel Y")

    def _set_noise_scale(self, e):
        try:
            self.noise_scale = float(e.data)
            print(f"[INFO] Noise scale set to {self.noise_scale}")
        except ValueError:
            print(f"[ERROR] Invalid noise scale: {e.data}")

    def _import_images(self, e):
        try:
            trim = lambda x: trim_image(x, (self.start_pixel_x, self.start_pixel_y), (self.end_pixel_x, self.end_pixel_y))
            self.volume = import_image_sequence(self.template, self.start_index, self.end_index, self.digit, self.format, processing=trim)
        except Exception as ex:
            print(f"[ERROR] {ex}")

    def _save_volume(self, e):
        self.file_picker_save_volume.on_result = self._save_volume_to_path
        self.file_picker_save_volume.save_file(
            dialog_title="Save Volume As",
            file_name="ct_volume.npy"
        )

    def _compute_orientations(self, e):
        try:
            if self.volume is None:
                raise ValueError("Volume must be imported first.")
            tensor = compute_structure_tesnsor(self.volume, self.noise_scale)
            self.theta, self.phi = compute_orientation(tensor)
            self.varphi = compute_orientation(tensor, reference_vector=[1, 0, 0])
            print("[SUCCESS] Orientations computed.")
        except Exception as ex:
            print(f"[ERROR] {ex}")

    def _compute_compressive_strength(self, e):
        if self.material_params is None:
            print("[ERROR] Material parameters not set.")
            return

        if self.volume is None:
            print("[ERROR] Volume not imported.")
            return

        try:
            self.UCS, self.UCstrain, self.sigma, self.eps = estimate_compression_strength_from_profile(self.varphi, self.material_params)
        except Exception as ex:
            print(f"[ERROR] {ex}")
    
    def _export_sscurve(self, e):
        if self.UCS is None:
            print("[ERROR] Compressive strength not computed.")
            return
        # stress series
        stress_series = pd.Series(self.sigma, name="Stress")
        strain_series = pd.Series(self.eps, name="Strain")

        try:
            df = pd.DataFrame([stress_series, strain_series], index=["Bin", "Histgram"]).transpose()
            self.file_picker_save_volume.on_result = lambda ev: self._save_csv_to_path(ev, df)
            self.file_picker_save_volume.save_file(
                dialog_title="Save SS curve Data As CSV",
                file_name="SS_cureve.csv"
            )
        except Exception as ex:
            print(f"[ERROR] Failed to prepare data: {ex}")

    def _export_histgram(self, e):
        if self.theta is None or self.phi is None or self.varphi is None:
            print("[ERROR] Orientation data not computed.")
            return

        try:
            df = compute_static_data(self.theta, self.phi, self.varphi, drop=int(self.noise_scale))
            self.file_picker_save_volume.on_result = lambda ev: self._save_csv_to_path(ev, df)
            self.file_picker_save_volume.save_file(
                dialog_title="Save Orientation Data As CSV",
                file_name="orientation_data.csv"
            )
        except Exception as ex:
            print(f"[ERROR] Failed to prepare data: {ex}")

    def _save_csv_to_path(self, e: ft.FilePickerResultEvent, df: pd.DataFrame):
        if e.path:
            try:
                df.to_csv(e.path, index=False)
                print(f"[SUCCESS] CSV saved to {e.path}")
            except Exception as ex:
                print(f"[ERROR] Failed to save CSV: {ex}")
        else:
            print("[INFO] Save cancelled.")

    def _apply_material_params(self, e):
        try:
            self.material_params = MaterialParams(
                longitudinal_modulus=float(self.material_inputs["longitudinal_modulus"].value),
                transverse_modulus=float(self.material_inputs["transverse_modulus"].value),
                poisson_ratio=float(self.material_inputs["poisson_ratio"].value),
                shear_modulus=float(self.material_inputs["shear_modulus"].value),
                tau_y=float(self.material_inputs["tau_y"].value),
                K=float(self.material_inputs["K"].value),
                n=float(self.material_inputs["n"].value)
            )
            print("[SUCCESS] Material parameters applied.")
        except ValueError as ex:
            print(f"[ERROR] Invalid material parameter: {ex}")

    def _parse_int(self, data, name, min_val=0, max_val=None):
        try:
            value = int(data)
            if value < min_val or (max_val is not None and value > max_val):
                raise ValueError
            return value
        except:
            print(f"[ERROR] Invalid {name}: {data}")
            raise

    def _save_volume_to_path(self, e: ft.FilePickerResultEvent):
        if self.volume is None:
            print("[ERROR] No volume to save.")
            return
        if e.path:
            np.save(e.path, self.volume)
            print(f"[SUCCESS] Volume saved to {e.path}")
        else:
            print("[INFO] Save cancelled.")

def main(page: ft.Page):
    page.title = "STA: Structure Tensor Analysis for composite structures"
    page.scroll = ft.ScrollMode.ALWAYS
    file_picker_file = ft.FilePicker()
    file_picker_file_save_volume = ft.FilePicker()
    page.overlay.append(file_picker_file)
    page.overlay.append(file_picker_file_save_volume)
    page.add(ft.Container(content=App(file_picker_file, file_picker_file_save_volume), width=1200, padding=20, alignment=ft.alignment.center))

ft.app(target=main)
