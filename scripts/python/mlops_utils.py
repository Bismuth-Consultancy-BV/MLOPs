import os
import hou
from hutil.Qt import QtCore, QtGui, QtWidgets
import subprocess

PIP_FOLDER = os.path.normpath(
    os.path.join(hou.text.expandString("$HOUDINI_USER_PREF_DIR"), "scripts", "python")
)


def generate_gpt_code_from_prompt(prompt, wrapper, model="gpt-3.5-turbo"):
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
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

def return_downloaded_checkpoints(subfolder="", replace_sign="-_-"):
    model_paths = ["$MLOPS_MODEL", "$MLOPS_MODEL"]
    root = hou.text.expandString("$MLOPS_MODELS")
    full_path = os.path.join(root, subfolder)
    for f in os.scandir(full_path):
        if f.is_dir():
            if f.name != "cache":
                model_paths.append(f.name.replace(replace_sign, "/"))
                model_paths.append(f.name.replace(replace_sign, "/"))
    return model_paths

def ensure_huggingface_model_local(model_name, model_path, cache_only=False):
    from huggingface_hub import snapshot_download
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import WEIGHTS_NAME, CONFIG_NAME, ONNX_WEIGHTS_NAME
    from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME

    path = hou.text.expandString(os.path.join(model_path, model_name.replace("/", "-_-")))
    cache = hou.text.expandString(os.path.join(model_path, "cache"))

    if os.path.isdir(model_name):
        return model_name
    if cache_only:
        return path.replace("\\", "/")

    model_name = model_name.replace("-_-", "/")
    allow_patterns = []
    try:
        config_dict = StableDiffusionPipeline.load_config(model_name, cache_dir=cache, resume_download=True, force_download=False)
        folder_names = [k for k in config_dict.keys() if not k.startswith("_")]
        allow_patterns += [os.path.join(k, "*") for k in folder_names]
        allow_patterns.append(StableDiffusionPipeline.config_name)
    except:
        pass
    allow_patterns += [WEIGHTS_NAME, SCHEDULER_CONFIG_NAME, CONFIG_NAME, ONNX_WEIGHTS_NAME, "*.json"]
    ignore_patterns = ["*.msgpack", "*.safetensors", "*.ckpt"]

    snapshot_download(repo_id=model_name, cache_dir=cache, local_dir=path, local_dir_use_symlinks=True, local_files_only=cache_only, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)
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


class MLOPSConvertModel(QtWidgets.QDialog):
    def __init__(self, parent):
        super(MLOPSConvertModel, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        self.buildUI()
        self.resize(self.minimumSizeHint())

    def closeEvent(self, event):
        pass

    def on_accept(self):
        from sdpipeline import convert_model
        model_name = self.model_name_field.text()
        checkpoint_file = hou.text.expandString(self.checkpoint_file_field.text())
        config_file = self.config_file_field.text()
        config_file = hou.text.expandString(config_file)
        checkpoint_file = hou.text.expandString(checkpoint_file)
        if config_file == "":
            config_file = None

        with hou.InterruptableOperation("Converting Model", open_interrupt_dialog=True) as operation:
            convert_model.convert(checkpoint_file, config_file, hou.text.expandString(os.path.join("$MLOPS_MODELS", model_name.replace("/", "-_-"))))

        hou.ui.displayMessage(f"You have successfully converted the {model_name} model!", buttons=('OK',), severity=hou.severityType.Message, title="MLOPs Plugin")

        self.close()

    def on_cancel(self):
        self.close()

    def open_checkpoint_dialog(self):

        directory = hou.ui.selectFile(title="MLOPs - Select Download Directory", file_type=hou.fileType.Any, multiple_select=False, pattern="*.safetensors *.ckpt", chooser_mode=hou.fileChooserMode.Read)
        if directory:
            self.checkpoint_file_field.setText(directory)

    def open_config_dialog(self):

        directory = hou.ui.selectFile(title="MLOPs - Select Download Directory", file_type=hou.fileType.Any, multiple_select=False, pattern="*.yaml", chooser_mode=hou.fileChooserMode.Read)
        if directory:
            self.config_file_field.setText(directory)

    def buildUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle("MLOPs - Convert Model")
        message_widget = QtWidgets.QLabel("Convert a checkpoint file to the Diffusers model used by this plugin!")
        layout.addWidget(message_widget)

        model_name_layout = QtWidgets.QHBoxLayout()
        model_path_label = QtWidgets.QLabel("Model Name: ")
        self.model_name_field = QtWidgets.QLineEdit("developer/model_name")
        model_name_layout.addWidget(model_path_label)
        model_name_layout.addWidget(self.model_name_field)

        checkpoint_file_layout = QtWidgets.QHBoxLayout()
        checkpoint_file_label = QtWidgets.QLabel("Checkpoint File: ")
        self.checkpoint_file_field = QtWidgets.QLineEdit()
        checkpoint_file_layout.addWidget(checkpoint_file_label)
        checkpoint_file_layout.addWidget(self.checkpoint_file_field)

        # Create QPushButton for directory browser
        self.checkpoint_file_button = QtWidgets.QPushButton("...")
        self.checkpoint_file_button.clicked.connect(self.open_checkpoint_dialog)
        checkpoint_file_layout.addWidget(self.checkpoint_file_button)

        layout.addLayout(model_name_layout)
        layout.addLayout(checkpoint_file_layout)


        config_file_layout = QtWidgets.QHBoxLayout()
        config_file_label = QtWidgets.QLabel("Config File: ")
        self.config_file_field = QtWidgets.QLineEdit()
        config_file_layout.addWidget(config_file_label)
        config_file_layout.addWidget(self.config_file_field)

        # Create QPushButton for directory browser
        self.config_file_button = QtWidgets.QPushButton("...")
        self.config_file_button.clicked.connect(self.open_config_dialog)
        config_file_layout.addWidget(self.config_file_button)
        layout.addLayout(config_file_layout)


        buttonlayout = QtWidgets.QHBoxLayout()
        self.update_button = QtWidgets.QPushButton("Convert")
        self.update_button.clicked.connect(self.on_accept)
        buttonlayout.addWidget(self.update_button)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel)
        buttonlayout.addWidget(self.cancel_button)

        layout.addLayout(buttonlayout)

        self.setMinimumSize(hou.ui.scaledSize(500), hou.ui.scaledSize(100))
        self.show()


class MLOPSPipInstall(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        self.buildUI()
        self.resize(self.minimumSizeHint())

    def closeEvent(self, event):
        pass

    def on_accept(self):
        user_input = self.dep_field.text()
        result = None
        if user_input:
            user_input = user_input.split(",")
            user_input = [x.strip() for x in user_input]
            result = user_input

        if not result:
            print("Canceled")
            self.close()
            return

        count = 0
        total = len(result)

        with hou.InterruptableOperation(
            "Pip Install", "Installing Dependencies", open_interrupt_dialog=True
        ) as operation:
            for dep in result:
                operation.updateLongProgress(
                    percentage=count / total, long_op_status=f"Installing {dep}"
                )
                pip_install(dep)
                count += 1

            # Informing user about the change
            hou.ui.displayMessage(
                "Install Complete",
                buttons=("OK",),
                severity=hou.severityType.Message,
                title="MLOPs Plugin",
            )

        self.close()

    def list_existing(self):
        try:
            from pip._vendor import pkg_resources

            dists = pkg_resources.find_on_path(None, PIP_FOLDER)
            dists = sorted(dists, key=lambda item: str(item))
            return dists
        except Exception as e:
            print(e, file=sys.stderr)
            return []

    def on_cancel(self):
        self.close()

    def buildUI(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.setWindowTitle("MLOPs - Pip Install")
        message_widget = QtWidgets.QLabel(
            "Enter a comma-separated list of dependencies"
        )
        layout.addWidget(message_widget)

        # the dependency input field
        input_layout = QtWidgets.QHBoxLayout()
        dep_label = QtWidgets.QLabel("Dependencies: ")
        self.dep_field = QtWidgets.QLineEdit("")
        input_layout.addWidget(dep_label)
        input_layout.addWidget(self.dep_field)

        layout.addLayout(input_layout)

        layout.addStretch(1)

        buttonlayout = QtWidgets.QHBoxLayout()
        install_button = QtWidgets.QPushButton("Install")
        install_button.clicked.connect(self.on_accept)
        buttonlayout.addWidget(install_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.on_cancel)
        buttonlayout.addWidget(cancel_button)

        # detail panel toggle
        self.detail_toggle = QtWidgets.QPushButton("Show Installed")
        self.detail_toggle.clicked.connect(self.toggle_detail_panel)
        buttonlayout.addWidget(self.detail_toggle)

        # closable detail panel for installed packages
        self.detail_panel = QtWidgets.QWidget()
        detail_panel_layout = QtWidgets.QVBoxLayout()

        detail_label = QtWidgets.QLabel("Installed Packages:")
        detail_panel_layout.addWidget(detail_label)

        scroll_area = QtWidgets.QScrollArea(self.detail_panel)
        scroll_area.setWidgetResizable(True)

        installed_label = QtWidgets.QLabel(
            "\n".join([str(x) for x in self.list_existing()])
        )

        scroll_area.setWidget(installed_label)
        detail_panel_layout.addWidget(scroll_area)

        self.detail_panel.setLayout(detail_panel_layout)
        self.detail_panel.hide()
        layout.addWidget(self.detail_panel)
        layout.addLayout(buttonlayout)

        self.setMinimumSize(hou.ui.scaledSize(300), hou.ui.scaledSize(150))
        self.show()

    def toggle_detail_panel(self):
        if self.detail_panel.isHidden():
            self.detail_panel.show()
            self.detail_toggle.setText("Hide Installed")

        else:
            self.detail_panel.hide()
            self.detail_toggle.setText("Show Installed")


def pip_install(dep):
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen(
        [
            "hython",
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            PIP_FOLDER,
            dep,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    while p.poll() is None:
        line = p.stdout.readline().decode().strip()
        if line:
            print(line)
    if p.returncode != 0:
        raise hou.Error("Installation failed.")
