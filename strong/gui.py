import flet as ft
import numpy as np
import sys
import os
import re
import pandas as pd
import strong as st
from flet.matplotlib_chart import MatplotlibChart
import matplotlib.pyplot as plt
import matplotlib
import json

matplotlib.use("svg")

# Hardcoded constants
TITLE = "STRONG"
VERSION = st.__version__
DESCRIPTION = "STRONG: Structure Tensor analysis of fiber Reinforced plastics " \
        "for cOmpressive streNGth simulation and digital twin development. This application is designed to analyze and visualize " \
        "the structure of composite materials using image sequences. " \
        "It allows users to import image sequences, compute orientations, estimate compressive strength, and generate 3D models of fibers. "
HEADER_1 = "Step 1: Import Image Sequences"
DESCRIPTION_1 = "Import image sequences from a folder" \
        "The application will automatically detect the file format and number of digits in the filenames. " \
        "Supported file types are: [dcm, png, tiff, tif] " \
        "Select the first image file to set the template for the sequence." \
        "Specify the range of frames to import and the pixel coordinates for cropping." \
        "The imported volume will be saved as a NumPy array with 'Save as npy button'." \
        "The npy file can be visualized using tomviz (https://tomviz.org/)."
HEADER_2 = "Step 2: Compute angles of fibers"
DESCRIPTION_2 = "Compute orientations from the imported image sequence. "\
                "This method can compute three metrics of angles: axial orientation (varphi, See 'a'), in-plane orientation (theta, See 'b'), "\
                "and out-of-plane orientation (phi, See 'c'). Determinations of each orientation are explaned in right figure. "\
                "Axial orientation is non-negative metric represents angle between the expected UD axis (z-axis) and "\
                "the local fiber. in-plane orientation is angle in xz-plane between the expected UD axis (z-axis) and "\
                "the local fiber. out-of-plane orientation is angle in yz-plane between the expected UD axis (z-axis) and "\
                "the local fiber. The computed orientations are saved as a CSV file with 'Export data' button. " \
                "The noise scale is used in the Gaussian filter to compute the structure tensor. " \
                "See details in (https://doi.org/10.1016/j.compositesa.2021.106541)."
HEADER_3 = "Step 3: Estimate Compressive strength"
DESCRIPTION_3 = "Estimate compressive strength from the computed orientations. This method considers the variation of fiber orientation."\
                "Details are described in (https://doi.org/10.1016/j.compositesa.2023.107821). " \
                "In this method, the compressive stress-strain curve is estimated as the superposition of weighted stress-strain curves with various "\
                "initial fiber misalignments. Ramberg-Osgood model is used to describe the shear non-linear behavior. There two modes to cosider the orientation: "\
                "In-plane mode and Axial orientation mode. In-plane mode is used only the in-plane orientation assuming the out-of-plane orientation is neglectable. "\
                "In this mode, the Average mode is selectable that neglects the average misalignment of CT-image. In this case, initial misalignment must be determined manually. "\
                "When the turn off the in-plane mode, the axial orientation mode is used. In this mode, the orienatation profile is directly used. "
HEADER_4 = "Step 4: Rebuild 3D model of fibers"
DESCRIPTION_4 = "Generate 3D model of fibers from the computed orientations. " \
                "The generated model is saved as an STL file with 'Export STL file' button. " \
                "The fiber positions are not correct because the fibers are initialized by Poisson disk sampling. " \
                "The fibers are oriented with the computed orientations, and contacting fibers are relaxed. " \
                "The diameter of fibers and fiber volume fraction are required. " \
                "The scaling factor is used to avoid contacting of fibers. " \
                "The step size is used to determine the length of resolution to fiber direction. " 


FILE_NAME_VOLUME = "volume.npy"
FILE_NAME_ORIENTATION_CSV = "orientation_data.csv"
FILE_NAME_SSCURVE = "streee_strain.csv"
FILE_NAME_FIBER = "fibers.stl"

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

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
        self._load_material_presets()

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
        self.average_mode_disabled = True

    def _build_ui(self):
        
        return [
            self._section_title(TITLE + " " + VERSION, DESCRIPTION),
            self._section_header(HEADER_1),
            self._section_description(DESCRIPTION_1),
            self._build_image_input_row(),
            self._build_crop_input_row(),
            self._build_import_buttons(),
            self._section_header(HEADER_2),
            self._section_description(DESCRIPTION_2, image="assets/orientation.tif"),
            self._build_orientation_row(),
            self._section_header(HEADER_3),
            self._section_description(DESCRIPTION_3),
            self._build_material_param_inputs(),
            self._build_compute_compressive_strength_button(),
            self._section_header(HEADER_4),
            self._section_description(DESCRIPTION_4),
            self._build_model_params_inputs(),
            self._build_modelconstruction_button()
        ]

    def _section_title(self, title, description):
        return ft.Row([
            ft.Container(ft.Image(src=resource_path("assets/icon.png"), width=180, height=180), 
                         padding=ft.padding.only(top=20, left=20, right=10)),
            ft.Column([
            ft.Container(content=ft.Text(title, theme_style=ft.TextThemeStyle.HEADLINE_LARGE)),
            ft.Container(
            content=ft.Text(description, size=14, color="gray"),
            width=600,
            padding=ft.padding.only(left=30, right=10))])])

    def _section_header(self, text):
        return ft.Column([
            ft.Divider(thickness=2, height=40),
            ft.Text(text, theme_style=ft.TextThemeStyle.HEADLINE_MEDIUM),
        ])

    def _section_description(self, description, image=None):
        if image is None:
            return ft.Column([ft.Container(
                content=ft.Text(description, size=14, color="gray"),
                width=1000,
                padding=ft.padding.only(left=30, right=10)),
            ])
        else:
            return ft.Row([ft.Container(
                content=ft.Text(description, size=14, color="gray"),
                width=500,
                padding=ft.padding.only(left=30, right=50)),
            ft.Image(src=resource_path(image), width=600)])

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
            ft.Image(src=resource_path("assets/crop_explanation.tif"), width=700)
        ])

    def _build_material_param_inputs(self):

        self.material_dropdown = ft.Dropdown(
        label="Material Preset",
        options=[ft.dropdown.Option(name) for name in self.material_presets.keys()],
        width=300,
        on_change=self._apply_material_preset
        )

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
                ft.Text(descriptions[key], width=200),
                self.material_inputs[key],
                ft.Text(units[key], width=100)
            ])
            param_rows.append(row)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("Stress-Strain Curve")
        ax.set_xlim(0, 0.01)
        ax.set_ylim(0, 2000)
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress [MPa]")
        ax.grid(True)
        fig.tight_layout()

        self.matplotlib_chart = MatplotlibChart(fig, expand=False)

        return ft.Row([
        ft.Container(
        content=ft.Column(
            controls=[self.material_dropdown] + param_rows,
            spacing=10
        ),
        padding=10,
        width=500
        ),
        ft.Container(
        content=self.matplotlib_chart,
        padding=50,
        width=600
        )
    ])
    
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
                ft.Text(descriptions[key], width=150),
                self.model_inputs[key],
                ft.Text(units[key], width=60)
            ])
            param_rows.append(row)

        return ft.Row([ft.Column(controls=param_rows, spacing=10),
                        ft.Image(src=resource_path("assets/dehom.tif"), width=750)])

    def _build_compute_compressive_strength_button(self):

        self.inplane_mode = ft.Switch(label="In-plane mode", on_change=self._change_average_mode_state, value=False)
        self.average_mode = ft.Switch(label="Average mode", value=False, disabled=True, on_change=self._change_textfield_state)
        self.manual_mode = ft.Switch(label="Manual mode", value=False, on_change=self._on_manual_mode_change)

        self.legend_field = ft.TextField(
            label="Legend",
            value="Case-1",
            width=250
        )

        self.standard_deviation_field = ft.TextField(
            label="Standard deviation",
            disabled=True,
            width=250
        )

        self.initial_misalignment_field = ft.TextField(
            label="Initial misalignment",
            disabled=True,
            width=250
        )

        return ft.Column([
            self.legend_field,
            self.initial_misalignment_field,
            self.standard_deviation_field,
            ft.Row([
                self.inplane_mode,
                self.average_mode,
                self.manual_mode,
            ]),
            ft.Row([
                ft.TextButton("Apply Parameters", on_click=self._apply_material_params, width=280),
                ft.TextButton("Compute Compressive Strength", on_click=self._compute_compressive_strength, width=280),
                ft.TextButton("Export SS-curve", on_click=self._export_sscurve, width=280)
            ])
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
            ft.TextButton("Export data", on_click=self._export_histgram, width=280),
            ft.TextButton("Save orientations as npy", on_click=self._export_orientations, width=280)
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

    def _change_average_mode_state(self, e: ft.ControlEvent):
        if self.inplane_mode.value:
            self.average_mode.disabled = False
        else:
            self.average_mode.disabled = True
            self.average_mode.value = False
        self.average_mode.update()
        self._change_textfield_state(None)

    def _change_textfield_state(self, e):
        self.initial_misalignment_field.disabled = not (self.inplane_mode.value and self.average_mode.value)
        self.initial_misalignment_field.update()

    def _set_noise_scale(self, e):
        try:
            self.noise_scale = int(e.data)
            print(f"[INFO] Noise scale set to {self.noise_scale}")
        except ValueError:
            print(f"[ERROR] Invalid noise scale: {e.data}")

    def _import_images(self, e):
        try:
            trim = lambda x: st.trim_image(x, (self.start_pixel_x, self.start_pixel_y), (self.end_pixel_x, self.end_pixel_y))
            self.volume = st.import_image_sequence(self.template, self.start_index, self.end_index, self.digit, self.format, processing=trim)
        except Exception as ex:
            print(f"[ERROR] {ex}")

    def _save_volume(self, e):
        self.file_picker_save_volume.on_result = self._save_volume_to_path
        self.file_picker_save_volume.save_file(
            dialog_title="Save Volume As",
            file_name=FILE_NAME_VOLUME
        )
        print(f"[SUCCESS] Volume saved to {e.path}")

    def _compute_orientations(self, e):
        print("[INFO] Take few minutes to compute orientations. Beginning...")
        try:
            if self.volume is None:
                raise ValueError("Volume must be imported first.")
            tensor = st.compute_structure_tensor(self.volume, self.noise_scale)
            self.theta, self.phi = st.compute_orientation(tensor)
            self.varphi = st.compute_orientation(tensor, reference_vector=[1, 0, 0])
            print("[SUCCESS] Orientations computed.")
        except Exception as ex:
            print(f"[ERROR] {ex}")

    def _compute_compressive_strength(self, e):
        if self.material_params is None:
            print("[ERROR] Material parameters not set.")
            return
        
        if self.manual_mode.value:
            mode_description = "Manual mode"
            print(f"[INFO] Current mode: {mode_description}")
            if self.standard_deviation_field.value == "":
                print("[ERROR] Standard deviation must be set.")
                return
            if self.initial_misalignment_field.value == "":
                print("[ERROR] Initial misalignment must be set.")
                return
            std_dev = float(self.standard_deviation_field.value)
            mean = float(self.initial_misalignment_field.value)
            try:
                self.UCS, self.UCstrain, self.sigma, self.eps = st.estimate_compression_strength(mean, std_dev, self.material_params)
                self.update_stress_strain_plot(self.eps, self.sigma)
            except Exception as ex:
                print(f"[ERROR] {ex}")
                return

        if self.volume is None:
            print("[ERROR] Volume not imported.")
            return

        if self.inplane_mode.value:
            mode_description = "In-plane mode"
            if self.average_mode.value:
                mode_description += " (Average mode)"
        else:
            mode_description = "Axial orientation mode"
        print(f"[INFO] Current mode: {mode_description}")

        std_dev = np.sqrt(np.var(self.theta.ravel()))
        mean = np.mean(self.theta.ravel())

        print(f"[INFO] Mean: {mean}, standart deviation: {std_dev}")

        # Axial orientation mode
        if not self.inplane_mode.value:

            try:
                self.UCS, self.UCstrain, self.sigma, self.eps = st.estimate_compression_strength_from_profile(self.varphi, self.material_params)
                self.update_stress_strain_plot(self.eps, self.sigma)
            except Exception as ex:
                print(f"[ERROR] {ex}")

        elif self.inplane_mode.value and not self.average_mode.value:

            try:
                self.UCS, self.UCstrain, self.sigma, self.eps = st.estimate_compression_strength(mean, std_dev,
                                                                                                 self.material_params)
                self.update_stress_strain_plot(self.eps, self.sigma)
            except Exception as ex:
                print(f"[ERROR] {ex}")
            
        # In-plane mode with average mode
        else:
            try:
                if self.initial_misalignment_field.value == "":
                    print("[ERROR] Initial misalignment must be set.")
                    return
                self.UCS, self.UCstrain, self.sigma, self.eps = st.estimate_compression_strength(float(self.initial_misalignment_field.value),
                                                                                                      std_dev, self.material_params)
                self.update_stress_strain_plot(self.eps, self.sigma)
            except Exception as ex:
                print(f"[ERROR] {ex}")

    def _model_construction(self, e):
        fibers = st.Fibers()
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
        fibers.initialize(self.volume.shape, float(self.fiber_diameter),
                          float(self.fiber_volume_fraction),
                          float(self.scale))
        step_size = int(self.step_size)
        print("[INFO] Model construction started.")
        for i in range(self.start_index, self.end_index, step_size):
            print(f"[INFO] Position: z={i}/{self.end_index}...")
            direction_x = step_size*np.tan(np.deg2rad(self.theta))[i]
            direction_y = step_size*np.tan(np.deg2rad(self.phi))[i]
            fibers.move_points(direction_x, direction_y)
            fibers.update_fiber(i, fibers.points)
        self.fibers_model = fibers
        print("[SUCCESS] Model construction completed.")

    def _export_stl(self, e):
        if self.fibers_model is None:
            print("[ERROR] Fibers model not constructed.")
            return
        try:
            mesh = st.generate_fiber_stl(self.fibers_model)
            self.file_picker_save_volume.on_result = lambda ev: self._save_mesh(ev, mesh)
            self.file_picker_save_volume.save_file(
                dialog_title="Save fibers as STL",
                file_name=FILE_NAME_FIBER
            )
        except Exception as ex:
            print(f"[ERROR] Failed to save fibers: {ex}")
    
    def _export_sscurve(self, e):
        if not hasattr(self, "stress_strain_history") or not self.stress_strain_history:
            print("[ERROR] No stress-strain data available.")
            return

        try:
            df_dict = {}
            for i, (strain, stress) in enumerate(self.stress_strain_history):
                label = self.ss_legend_labels[i] if hasattr(self, "ss_legend_labels") else f"Case-{i+1}"
                df_dict[f"Strain_{label}"] = strain
                df_dict[f"Stress_{label}"] = stress
            df = pd.DataFrame(df_dict)

            self.file_picker_save_volume.on_result = lambda ev: self._save_csv_to_path(ev, df)
            self.file_picker_save_volume.save_file(
                dialog_title="Save SS curve Data As CSV",
                file_name=FILE_NAME_SSCURVE
            )
        except Exception as ex:
            print(f"[ERROR] Failed to prepare data: {ex}")

    def _export_histgram(self, e):
        if self.theta is None or self.phi is None or self.varphi is None:
            print("[ERROR] Orientation data not computed.")
            return

        try:
            df = st.compute_static_data(self.theta, self.phi, self.varphi, drop=int(self.noise_scale))
            self.file_picker_save_volume.on_result = lambda ev: self._save_csv_to_path(ev, df)
            self.file_picker_save_volume.save_file(
                dialog_title="Save Orientation Data As CSV",
                file_name="orientation_data.csv"
            )
        except Exception as ex:
            print(f"[ERROR] Failed to prepare data: {ex}")

    def _export_orientations(self, e):
        import tempfile
        import zipfile

        if self.theta is None or self.phi is None or self.varphi is None:
            print("[ERROR] Orientation data not computed.")
            return

        try:
            tmpdir = tempfile.gettempdir()
            theta_path = os.path.join(tmpdir, "theta.npy")
            phi_path = os.path.join(tmpdir, "phi.npy")
            varphi_path = os.path.join(tmpdir, "varphi.npy")
            zip_path = os.path.join(tmpdir, "orientations.zip")

            np.save(theta_path, self.theta)
            np.save(phi_path, self.phi)
            np.save(varphi_path, self.varphi)

            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(theta_path, arcname="theta.npy")
                zipf.write(phi_path, arcname="phi.npy")
                zipf.write(varphi_path, arcname="varphi.npy")

            # 中間ファイル削除
            for f in [theta_path, phi_path, varphi_path]:
                try:
                    os.remove(f)
                except Exception as rm_ex:
                    print(f"[WARNING] Failed to delete temp file {f}: {rm_ex}")

            # エクスプローラーで保存先選択
            self.file_picker_save_volume.on_result = lambda ev: self._save_zip(ev, zip_path)
            self.file_picker_save_volume.save_file(
                dialog_title="Save Orientation ZIP",
                file_name="orientations.zip"
            )
        except Exception as ex:
            print(f"[ERROR] Failed during export: {ex}")
            
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

    def _save_zip(self, e: ft.FilePickerResultEvent, source_zip_path):
        if e.path:
            try:
                import shutil
                shutil.copy(source_zip_path, e.path)
                print(f"[SUCCESS] ZIP saved to {e.path}")
            except Exception as ex:
                print(f"[ERROR] Failed to save ZIP: {ex}")
        else:
            print("[INFO] Save cancelled.")

    def _apply_material_params(self, e):
        try:
            self.material_params = st.MaterialParams(
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

    def _apply_material_preset(self, e: ft.ControlEvent):
        preset_name = e.control.value
        if preset_name == "Custom":
            print("[INFO] Custom preset selected. Please enter parameters manually.")
            return

        if preset_name in self.material_presets:
            preset = self.material_presets[preset_name]
            for key, value in preset.items():
                self.material_inputs[key].value = str(value)
                self.material_inputs[key].update()
            print(f"[INFO] Material preset '{preset_name}' applied.")

    def _apply_model_params(self, e):
        try:
            self.fiber_diameter = float(self.model_inputs["fiber_diameter"].value)
            self.fiber_volume_fraction = float(self.model_inputs["fiber_volume_fraction"].value)
            self.scale = float(self.model_inputs["scale"].value)
            self.step_size = int(self.model_inputs["step_size"].value)
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
            # Swap axis
            swapped_volume = np.swapaxes(self.volume, 0, 2)
            np.save(e.path, swapped_volume)
            print(f"[SUCCESS] Volume saved to {e.path}")
        else:
            print("[INFO] Save cancelled.")

    def update_stress_strain_plot(self, strain, stress):
        if not hasattr(self, "stress_strain_history"):
            self.stress_strain_history = []
            self.ss_legend_labels = []

        legend_label = self.legend_field.value.strip() or "Case-1"
        base_label = legend_label
        count = 1
        while legend_label in self.ss_legend_labels:
            count += 1
            legend_label = f"{base_label} ({count})"

        self.stress_strain_history.append((strain, stress))
        self.ss_legend_labels.append(legend_label)

        fig, ax = plt.subplots(figsize=(4, 3))
        for (s, t), label in zip(self.stress_strain_history, self.ss_legend_labels):
            ax.plot(s, t, label=label)

        ax.set_title("Stress-Strain Curve")
        ax.set_xlim(0, 0.01)
        ax.set_ylim(0, 2000)
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress [MPa]")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        self.matplotlib_chart.figure = fig
        self.matplotlib_chart.update()

    def _load_material_presets(self):
        path = resource_path(r"assets\material_params.json")
        try:
            with open(path, "r") as f:
                self.material_presets = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load material presets: {e}")
            self.material_presets = {}

        self.material_presets["Custom"] = {}

    def _on_manual_mode_change(self, e):
        if self.manual_mode.value:
            # Manual mode ON
            self.inplane_mode.disabled = True
            self.inplane_mode.value = False
            self.inplane_mode.update()

            self.average_mode.disabled = True
            self.average_mode.value = False
            self.average_mode.update()

            self.initial_misalignment_field.disabled = False
            self.standard_deviation_field.disabled = False
        else:
            # Manual mode OFF
            self.inplane_mode.disabled = False
            self.inplane_mode.update()

            self._change_average_mode_state(None)  # this will update average_mode + misalignment state

            self.standard_deviation_field.disabled = True
            self.standard_deviation_field.update()
        self.initial_misalignment_field.update()
        self.standard_deviation_field.update()

def main(page: ft.Page):
    
    page.title = "STRONG" + " " + VERSION
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
            ft.Container(
                content=ft.ListView(
                    controls=app.controls,
                    expand=True,
                    spacing=30,
                    padding=ft.padding.only(left=10)
                ),
                expand=True
            ),
            #ft.Divider(thickness=2, height=10, color="black"),
            #ft.Text("Console Output", size=20, text_align=ft.TextAlign.CENTER),
            ft.Container(
                content=app.console_output,
                height=90,
                bgcolor="#747474",
                padding=ft.padding.only(left=20, top=20),
            ),
        ], expand=True)
    )
    print("[INFO] This is console that shows the outputs from this application.")
ft.app(target=main)