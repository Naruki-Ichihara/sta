import flet as ft
import os
import shutil
import re


def main(page: ft.Page):
    page.title = "画像連番リネームツール"
    page.window_width = 800
    page.window_height = 600

    folder_path = ft.TextField(label="フォルダパス", read_only=True, expand=True)
    base_name = ft.TextField(label="新しいベース名", hint_text="例: sample")
    digit_length = ft.TextField(label="連番の桁数", hint_text="例: 4", value="3", width=100)
    preview_list = ft.ListView(expand=True, spacing=5, padding=10)

    image_files = []

    def result_handler(e2: ft.FilePickerResultEvent):
        if e2.path:
            folder_path.value = e2.path
            load_images(e2.path)
            page.update()

    def pick_folder(e):
        file_picker.get_directory_path()

    def load_images(path):
        nonlocal image_files
        image_files = [f for f in sorted(os.listdir(path)) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
        update_preview()

    def extract_base(filename):
        # 数字だけ取り除く、または_数字を除去
        base = re.sub(r'[_\-]?[0-9]+$', '', os.path.splitext(filename)[0])
        return base

    def update_preview(e=None):
        preview_list.controls.clear()
        try:
            digits = int(digit_length.value)
        except ValueError:
            digits = 3

        for idx, fname in enumerate(image_files, 1):
            ext = os.path.splitext(fname)[1]
            new_name = f"{base_name.value}_{str(idx).zfill(digits)}{ext}"
            preview_list.controls.append(ft.Text(f"{fname} -> {new_name}"))
        page.update()

    def rename_files(e):
        if not folder_path.value or not base_name.value:
            page.dialog = ft.AlertDialog(title=ft.Text("エラー"), content=ft.Text("フォルダとベース名を入力してください"))
            page.dialog.open = True
            page.update()
            return

        try:
            digits = int(digit_length.value)
        except ValueError:
            digits = 3

        try:
            for idx, fname in enumerate(image_files, 1):
                ext = os.path.splitext(fname)[1]
                new_name = f"{base_name.value}_{str(idx).zfill(digits)}{ext}"
                src = os.path.join(folder_path.value, fname)
                dst = os.path.join(folder_path.value, new_name)
                shutil.move(src, dst)

            load_images(folder_path.value)
            page.dialog = ft.AlertDialog(title=ft.Text("完了"), content=ft.Text("リネームが完了しました！"))
            page.dialog.open = True
            page.update()
        except Exception as ex:
            page.dialog = ft.AlertDialog(title=ft.Text("エラー"), content=ft.Text(str(ex)))
            page.dialog.open = True
            page.update()

    file_picker = ft.FilePicker(on_result=result_handler)
    page.overlay.append(file_picker)

    page.add(
        ft.Row([
            folder_path,
            ft.IconButton(icon=ft.icons.FOLDER_OPEN, on_click=pick_folder)
        ], spacing=10),
        base_name,
        digit_length,
        ft.Row([
            ft.ElevatedButton("プレビュー更新", on_click=update_preview),
            ft.ElevatedButton("リネーム実行", on_click=rename_files, color="white", bgcolor="blue")
        ], spacing=10),
        ft.Divider(),
        preview_list
    )


if __name__ == "__main__":
    ft.app(target=main)