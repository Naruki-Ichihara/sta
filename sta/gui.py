import flet as ft
import numpy as np
import sys
import re
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sta.io import import_image_sequence, trim_image
from sta.analysis import compute_structure_tesnsor, compute_orientation, compute_static_data
from sta.simulation import MaterialParams, estimate_compression_strength_from_profile
from sta.dehom import Fibers, generate_fiber_stl
from flet.matplotlib_chart import MatplotlibChart

plt.rcParams['font.family'] = 'Sans'
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 10)))
matplotlib.use("svg")


class ConsoleOutput:
    def __init__(self, list_view_control):
        self.list_view_control = list_view_control

    def write(self, message):
        lines = str(message).splitlines()
        for line in lines:
            self.list_view_control.controls.append(
                ft.Text(line, size=12, color="white")
            )
        self.list_view_control.update()

    def flush(self):
        pass

class App:
    def __init__(self, file_picker_file, file_picker_save_volume):
        self.file_picker_file = file_picker_file
        self.file_picker_save_volume = file_picker_save_volume

        self.console_output = ft.ListView(
            controls=[],
            auto_scroll=True,
            expand=True,
            spacing=1
            )
        sys.stdout = ConsoleOutput(self.console_output)

        self.template_field = ft.TextField(hint_text="Path template", read_only=True, width=700)
        self.digit_field = ft.TextField(hint_text="Digit", read_only=True, width=100)
        self.format_field = ft.TextField(hint_text="Format", read_only=True, width=100)

        self._init_state()
        self.controls = self._build_ui()

        self.UCS = None
        self.UCstrain = None
        self.sigma = None
        self.eps = None

    def _init_state(self):
        fig, ax = plt.subplots()
        self.fibers_model = None
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
        self.fiber_diameter = None
        self.fiber_volume_fraction = None
        self.scale = None
        self.step_size = None
        self.chart = (MatplotlibChart(fig, expand=True))

    def _build_ui(self):
        return [
            self._section_header("Step 1: Import Image Sequences."),
            self._build_image_input_row(),
            self._build_crop_input_row(),
            self._build_import_buttons(),
            self._section_header("Step 2: Compute orientations."),
            self._build_orientation_row(),
            self._section_header("Step 3: Estimate Compressive strength."),
            self._build_material_param_inputs(),
            self._build_compute_compressive_strength_button(),
            self._section_header("Step 4: Rebuild 3D model of fibers."),
            self._build_model_params_inputs(),
            self._build_modelconstruction_button()
        ]

    def _section_header(self, text):
        return ft.Column([
            ft.Divider(thickness=2, height=50),
            ft.Text(text, theme_style=ft.TextThemeStyle.HEADLINE_MEDIUM),
            ft.Divider(thickness=2, height=50),
        ])

    def _build_image_input_row(self):
        return ft.Row([
            ft.TextButton("Select first image", on_click=self._select_file, width=160),
            self.template_field,
            self.digit_field,
            self.format_field
        ])

    def _build_crop_input_row(self):
        self.crop_inputs = {
            "start_index": ft.TextField(width=100),
            "end_index": ft.TextField(width=100),
            "start_pixel_x": ft.TextField(width=100),
            "start_pixel_y": ft.TextField(width=100),
            "end_pixel_x": ft.TextField(width=100),
            "end_pixel_y": ft.TextField(width=100)
        }

        descriptions = {
        "start_index": "Frame number to start importing",
        "end_index": "Frame number to stop importing",
        "start_pixel_x": "Start pixel X-coordinate for cropping",
        "start_pixel_y": "Start pixel Y-coordinate for cropping",
        "end_pixel_x": "End pixel X-coordinate for cropping",
        "end_pixel_y": "End pixel Y-coordinate for cropping"
        }

        param_rows = []
        for key in self.crop_inputs.keys():
            row = ft.Row([
                ft.Text(descriptions[key], width=300),
                self.crop_inputs[key]
            ])
            param_rows.append(row)

        return ft.Row([
            ft.Column(controls=param_rows, spacing=10, alignment=ft.MainAxisAlignment.CENTER),
            ft.Image(src="images\crop_explanation.tif", width=700)
        ])


    def _build_orientation_row(self):
        return ft.Row([
            ft.TextField(hint_text="Noise scale", on_submit=self._set_noise_scale, width=180),
            ft.TextButton("Compute orientations", on_click=self._compute_orientations, width=280),
            ft.TextButton("Export data", on_click=self._export_histgram, width=280)
        ])

    def _build_material_param_inputs(self):
        self.material_inputs = {
            "longitudinal_modulus": ft.TextField(width=150),
            "transverse_modulus": ft.TextField(width=150),
            "poisson_ratio": ft.TextField(width=150),
            "shear_modulus": ft.TextField(width=150),
            "tau_y": ft.TextField(width=150),
            "K": ft.TextField(width=150),
            "n": ft.TextField(width=150)
        }

        units = {
            "longitudinal_modulus": "[MPa]",
            "transverse_modulus": "[MPa]",
            "poisson_ratio": "[-]",
            "shear_modulus": "[MPa]",
            "tau_y": "[MPa]",
            "K": "[MPa]",
            "n": "[-]"
        }

        descriptions = {
            "longitudinal_modulus": "Longitudinal Modulus (E1)",
            "transverse_modulus": "Transverse Modulus (E2)",
            "poisson_ratio": "Poisson Ratio (v12)",
            "shear_modulus": "Shear Modulus (G12)",
            "tau_y": "Shear Yield Stress (τy)",
            "K": "Hardening Coefficient (K)",
            "n": "Hardening Exponent (n)"
        }

        param_rows = []
        for key in self.material_inputs.keys():
            row = ft.Row([
                ft.Text(descriptions[key], width=300),
                self.material_inputs[key],
                ft.Text(units[key], width=100)
            ])
            param_rows.append(row)

        return ft.Row([
            ft.Column(controls=param_rows, spacing=10),
            self.chart])
    
    def _build_model_params_inputs(self):
        self.model_inputs = {
            "fiber_diameter": ft.TextField(width=150),
            "fiber_volume_fraction": ft.TextField(width=150),
            "scale": ft.TextField(width=150),
            "step_size": ft.TextField(width=150)
        }

        units = {
            "fiber_diameter": "[px]",
            "fiber_volume_fraction": "[-]",
            "scale": "[-]",
            "step_size": "[px]"
        }

        descriptions = {
            "fiber_diameter": "Diameter of each fibers (Typically 7um)",
            "fiber_volume_fraction": "Volume fration of fiber contents",
            "scale": "Scaleing of fibers, to avoid contacting of fibers",
            "step_size": "Length of resolution to fiber direction."
        }

        param_rows = []
        for key in self.model_inputs.keys():
            row = ft.Row([
                ft.Text(descriptions[key], width=300),
                self.model_inputs[key],
                ft.Text(units[key], width=100)
            ])
            param_rows.append(row)

        return ft.Column(controls=param_rows, spacing=10)

    def _build_compute_compressive_strength_button(self):
        return ft.Row([
            ft.TextButton("Apply Parameters", on_click=self._apply_material_params, width=280),
            ft.TextButton("Compute Compressive Strength", on_click=self._compute_compressive_strength, width=280),
            ft.TextButton("Export SS-curve", on_click=self._export_sscurve, width=280)
        ])

    def _build_image_input_row(self):
        return ft.Row([
            ft.TextButton("Select first image", on_click=self._select_file, width=160),
            self.template_field,
            self.digit_field,
            self.format_field
        ])


    def _build_import_buttons(self):
        return ft.Row([
            ft.TextButton("Apply parameters", on_click=self._apply_crop_params, width=280),
            ft.TextButton("Import image sequence", on_click=self._import_images, width=280),
            ft.TextButton("Save volume as npy", on_click=self._save_volume, width=280)
        ])

    def _build_orientation_row(self):
        return ft.Row([
            ft.TextField(hint_text="Noise scale", on_submit=self._set_noise_scale, width=180),
            ft.TextButton("Compute orientations", on_click=self._compute_orientations, width=280),
            ft.TextButton("Export data", on_click=self._export_histgram, width=280)
        ])
    
    def _build_modelconstruction_button(self):
        return ft.Row([
            ft.TextButton("Apply Parameters", on_click=self._apply_model_params, width=280),
            ft.TextButton("Generate stl file", on_click=self._model_construction, width=280),
            ft.TextButton("Export STL file", on_click=self._export_stl, width=280)
        ])

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

    def _set_noise_scale(self, e):
        try:
            self.noise_scale = int(e.data)
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
        print(f"[SUCCESS] Volume saved to {e.path}")

    def _compute_orientations(self, e):
        print("[INFO] Take few minutes to compute orientations. Beginning...")
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

        fig, ax = plt.subplots()
        ax.set_xlabel("Axial compressive strain [-]")
        ax.set_ylabel("Axial compressive stress [MPa]")
        ax.plot(self.eps, self.sigma, label="Stress-Strain Curve")
        self.chart = MatplotlibChart(fig, expand=True, height=400, width=800)

    def _model_construction(self, e):
        fibers = Fibers()
        if self.volume is None:
            print("[ERROR] Volume not imported.")
            return
        if self.fiber_diameter is None:
            print("[ERROR] Fiber diameter must be set.")
        if self.fiber_volume_fraction is None:
            print("[ERROR] Fiber volume fraction must be set.")
            return
        if self.scale is None:
            print("[ERROR] Scale must be set.")
        if self.step_size is None:
            print(f"[ERROR] step_size must be set.")
        fibers.initialize(self.volume.shape, float(self.fiber_diameter[0]),
                          float(self.fiber_volume_fraction[0]),
                          float(self.scale[0]))
        step_size = int(self.step_size[0])
        print("[INFO] Model construction started.")
        for i in range(self.start_index, self.end_index, step_size):
            print(f"[INFO] Position: z={i}/{self.end_index}...")
            direction_x = step_size*np.tan(np.deg2rad(self.theta))[i]
            direction_y = step_size*np.tan(np.deg2rad(self.phi))[i]
            fibers.move_points(direction_x, direction_y)
            fibers.update_fiber(i, fibers.points)
        self.fibers_model = fibers
        print("[SUCCESS] Model construction completed.")
        pass

    def _export_stl(self, e):
        if self.fibers_model is None:
            print("[ERROR] Fibers model not constructed.")
            return
        try:
            mesh = generate_fiber_stl(self.fibers_model)
            self.file_picker_save_volume.on_result = lambda ev: self._save_mesh(ev, mesh)
            self.file_picker_save_volume.save_file(
                dialog_title="Save fibers as STL",
                file_name="fibers.stl"
            )
        except Exception as ex:
            print(f"[ERROR] Failed to save fibers: {ex}")
    
    def _export_sscurve(self, e):
        if self.UCS is None:
            print("[ERROR] Compressive strength not computed.")
            return
        # stress series
        stress_series = pd.Series(self.sigma, name="Stress")
        strain_series = pd.Series(self.eps, name="Strain")

        try:
            df = pd.DataFrame([stress_series, strain_series], index=["Stress", "Strain"]).transpose()
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

    def _save_mesh(self, e: ft.FilePickerResultEvent, mesh):
        
        if e.path:
            try:
                mesh.save(e.path)
                print(f"[SUCCESS] Mesh saved to {e.path}")
            except Exception as ex:
                print(f"[ERROR] Failed to save mesh: {ex}")
        else:
            print("[INFO] Save cancelled.")

    def _apply_crop_params(self, e):
        try:
            self.start_index = self._parse_int(self.crop_inputs["start_index"].value, "Start index")
            self.end_index = self._parse_int(self.crop_inputs["end_index"].value, "End index")
            self.start_pixel_x = self._parse_int(self.crop_inputs["start_pixel_x"].value, "Start Pixel X")
            self.start_pixel_y = self._parse_int(self.crop_inputs["start_pixel_y"].value, "Start Pixel Y")
            self.end_pixel_x = self._parse_int(self.crop_inputs["end_pixel_x"].value, "End Pixel X")
            self.end_pixel_y = self._parse_int(self.crop_inputs["end_pixel_y"].value, "End Pixel Y")
            print(f"[INFO] Parameters applied.")
        except ValueError as ex:
            print(f"[ERROR] Invalid crop parameter: {ex}") 

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

    def _apply_model_params(self, e):
        try:
            self.fiber_diameter=self.model_inputs["fiber_diameter"].value,
            self.fiber_volume_fraction=self.model_inputs["fiber_volume_fraction"].value,
            self.scale=self.model_inputs["scale"].value,
            self.step_size=self.model_inputs["step_size"].value
            print("[SUCCESS] Model parameters applied.")
        except ValueError as ex:
            print(f"[ERROR] Invalid model parameter: {ex}")

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
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 1280
    page.window_height = 2000
    file_picker_file = ft.FilePicker()
    file_picker_file_save_volume = ft.FilePicker()
    page.overlay.append(file_picker_file)
    page.overlay.append(file_picker_file_save_volume)
    app = App(file_picker_file, file_picker_file_save_volume)
    app.page = page

    page.add(
            ft.Column([
            ft.Text("Console Output", size=20, text_align=ft.TextAlign.CENTER),
            ft.Container(
                content=app.console_output,
                height=100,
                bgcolor="#333333",
                padding=ft.padding.only(left=20),
            ),
            ft.Container(
                content=ft.ListView(
                    controls=app.controls,
                    expand=True,
                    spacing=30,
                    padding=ft.padding.only(left=10)
                ),
                expand=True
            )
        ], expand=True)
    )

ft.app(target=main)