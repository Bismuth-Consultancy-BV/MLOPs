import os
import openai
import hou
from hutil.Qt import QtCore, QtGui, QtWidgets
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_gpt_code_from_prompt(prompt, wrapper, model="gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": f"{wrapper} {prompt}"}
        ]
    )
    return completion.choices[0].message.content

def is_relevant_parm(kwargs, parmtype):
    if parmtype == "wrangle":
        if len(kwargs["parms"]) == 0:
            return False
    return True

def return_downloaded_checkpoints():
    model_paths = ["$MLOPS_MODEL", "$MLOPS_MODEL"]
    root = hou.text.expandString("$MLOPS_MODELS")
    for f in os.scandir(root):
        if f.is_dir():
            if f.name != "cache":
                model_paths.append(f.name.replace("-_-", "/"))
                model_paths.append(f.name.replace("-_-", "/"))
    return model_paths

def ensure_huggingface_model_local(model_name, model_path, cache_only=False):
    path = hou.text.expandString(os.path.join(model_path, model_name.replace("/", "-_-")))
    cache = hou.text.expandString(os.path.join(model_path, "cache"))

    # print("ssss", model_name)
    if os.path.isdir(model_name):
        # print("name", model_name)
        return model_name
    if cache_only:
        # print("cache", path)
        return path.replace("\\", "/")

    from huggingface_hub import snapshot_download
    model_name = model_name.replace("-_-", "/")
    # print("download", model_name)
    snapshot_download(repo_id=model_name, cache_dir=cache, local_dir=path, repo_type="model", local_dir_use_symlinks=True, local_files_only=cache_only)
    return path.replace("\\", "/")


class MLOPSCheckpointDownloader(QtWidgets.QDialog):
    def __init__(self, parent):
        super(MLOPSCheckpointDownloader, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        self.buildUI()
        self.resize(self.minimumSizeHint())

    def closeEvent(self, event):
        pass

    def on_accept(self):
        model_name = self.model_path_field.text()
        download_dir = hou.text.expandString(self.download_directory_field.text())

        ensure_huggingface_model_local(model_name, download_dir)
        hou.ui.displayMessage(f"You have successfully downloaded or updated the {model_name} model!", buttons=('OK',), severity=hou.severityType.Message, title="MLOPs Plugin")

        self.close()

    def on_cancel(self):
        self.close()

    def open_directory_dialog(self):

        directory = hou.ui.selectFile(title="MLOPs - Select Download Directory", file_type=hou.fileType.Directory, multiple_select=False, chooser_mode=hou.fileChooserMode.Read)
        if directory:
            self.download_directory_field.setText(directory)

    def buildUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle("MLOPs - Model Download")
        message_widget = QtWidgets.QLabel("Automatically download a checkpoint by name from the Huggingface Hub!")
        layout.addWidget(message_widget)

        layout_model = QtWidgets.QHBoxLayout()
        model_path_label = QtWidgets.QLabel("Model Name: ")
        self.model_path_field = QtWidgets.QLineEdit("stabilityai/stable-diffusion-2-1")
        layout_model.addWidget(model_path_label)
        layout_model.addWidget(self.model_path_field)

        layout_browse = QtWidgets.QHBoxLayout()
        download_path_label = QtWidgets.QLabel("Download Directory: ")
        self.download_directory_field = QtWidgets.QLineEdit("$MLOPS_MODELS")
        layout_browse.addWidget(download_path_label)
        layout_browse.addWidget(self.download_directory_field)

        # Create QPushButton for directory browser
        self.dir_button = QtWidgets.QPushButton("...")
        self.dir_button.clicked.connect(self.open_directory_dialog)
        layout_browse.addWidget(self.dir_button)

        layout.addLayout(layout_model)
        layout.addLayout(layout_browse)

        buttonlayout = QtWidgets.QHBoxLayout()
        self.update_button = QtWidgets.QPushButton("Download")
        self.update_button.clicked.connect(self.on_accept)
        buttonlayout.addWidget(self.update_button)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel)
        buttonlayout.addWidget(self.cancel_button)

        layout.addLayout(buttonlayout)

        self.setMinimumSize(hou.ui.scaledSize(500), hou.ui.scaledSize(100))
        self.show()
