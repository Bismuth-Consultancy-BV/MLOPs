import hashlib
import json
import os
import shutil
import ssl
import subprocess
from urllib import request

import hou
from hutil.Qt import QtCore, QtGui, QtWidgets

PIP_FOLDER = os.path.normpath(
    os.path.join(hou.text.expandString("$MLOPS"), "data", "dependencies", "python")
)


def install_mlops_dependencies():
    FOLDER = os.path.normpath(
        os.path.join(hou.text.expandString("$HOUDINI_TEMP_DIR"), "MLOPs")
    )
    PIPINSTALLFILE = os.path.normpath(os.path.join(FOLDER, "get-pip.py"))

    check_mlops_version()

    # Downloading pip installation file
    if not os.path.isdir(FOLDER):
        os.makedirs(FOLDER)

    total = 3
    count = 1
    with hou.InterruptableOperation(
        "Installing Dependencies, downloading ~6Gb", open_interrupt_dialog=True
    ) as operation:
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW

        env = os.environ.copy()
        if "PYTHONPATH" in env:
            del env["PYTHONPATH"]

        # Trying to download get-pip with curl , this can quietly fail when behind proxy
        hou.ui.setStatusMessage("curl download get-pip.py")
        p = subprocess.Popen(
            ["curl", "-o", PIPINSTALLFILE, "https://bootstrap.pypa.io/get-pip.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        out, err = p.communicate()
        if err:
            raise hou.Error(out.decode())

        # Checking if get-pip downloaded with curl , otherwise try download previous get-pip using urllib with SSL certification and test Md5 hash for safety
        if not os.path.isfile(PIPINSTALLFILE):
            hou.ui.setStatusMessage(
                "curl failed to download get-pip.py , trying urllib."
            )
            remote_url = "https://github.com/pypa/get-pip/raw/2b873b978dbfdfc5e15ef5c3adf4354084612432/public/get-pip.py"  # last known get-pip.py from 20230422
            # Create an SSL context with certificate verification
            context = ssl.create_default_context()
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            with request.urlopen(remote_url, context=context) as response:
                with open(PIPINSTALLFILE, "wb") as file:
                    while True:
                        chunk = response.read(1024)
                        if not chunk:
                            break
                        file.write(chunk)
            if os.path.isfile(PIPINSTALLFILE):
                hou.ui.setStatusMessage("testing get-pip.py MD5.")
                safe_md5 = "6f33e0cffbbd2093f2406f8d0839b01f"  # last known get-pip.py MD5 hash from 20230422
                test_md5 = hashlib.md5(open(PIPINSTALLFILE, "rb").read()).hexdigest()
                if safe_md5 != test_md5:
                    os.remove(
                        PIPINSTALLFILE
                    )  # delete the unsafe get-pip.py if doesnt match
                    raise hou.Error(
                        "unsafe get-pip.py downloaded with incorrect MD5hash. safe_MD5 = "
                        + safe_md5
                        + " != "
                        + test_md5
                    )

        # If that also fails to download , then error out and ask for manual download
        if not os.path.isfile(PIPINSTALLFILE):
            raise hou.Error(
                "Failed to download get-pip.py , please download get-pip.py manually from https://bootstrap.pypa.io/ , and place in here = "
                + PIPINSTALLFILE
            )
        operation.updateProgress(percentage=count / total)
        count += 1

        # Installing pip to Houdini
        hou.ui.setStatusMessage("Installing pip to Houdini")
        p = subprocess.Popen(
            ["hython", PIPINSTALLFILE],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        out, err = p.communicate()
        if err:
            raise hou.Error(out.decode())
        operation.updateProgress(percentage=count / total)
        count += 1

        dependencies_dir = os.path.normpath(
            os.path.join(
                hou.text.expandString("$MLOPS"), "data", "dependencies", "python"
            )
        )
        try:
            if os.path.isdir(dependencies_dir):
                shutil.rmtree(dependencies_dir)
        except:
            raise hou.Error(
                f"Please close Houdini and delete {dependencies_dir}\nAfter deleting the specified folder restart Houdini and install dependencies again."
            )

        hou.ui.setStatusMessage("Installing requirements.txt")
        pip_install(hou.text.expandString("$MLOPS/requirements.txt"), True, True)
        count += 1

    # Informing user about the change
    hou.ui.displayMessage(
        "You have now installed the required dependencies for this build of Houdini.\nPlease restart Houdini for it to take effect",
        buttons=("OK",),
        severity=hou.severityType.Message,
        title="MLOPs Plugin",
    )


def generate_gpt_code_from_prompt(prompt, wrapper, model="gpt-3.5-turbo"):
    import openai

    openai.api_key = os.getenv("OPENAI_API_KEY")
    completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": f"{wrapper} {prompt}"}]
    )
    return completion.choices[0].message.content


def is_relevant_parm(kwargs, parmtype):
    if parmtype == "wrangle":
        if len(kwargs["parms"]) == 0:
            return False
    return True


def return_downloaded_checkpoints(
    root="$MLOPS_MODELS", subfolder="", replace_sign="-_-"
):
    model_paths = ["$MLOPS_SD_MODEL", "$MLOPS_SD_MODEL"]
    root = hou.text.expandString(root)
    full_path = os.path.join(root, subfolder)
    if os.path.isdir(full_path):
        for f in os.scandir(full_path):
            if f.is_dir():
                if f.name != "cache":
                    model_paths.append(f.name.replace(replace_sign, "/"))
                    model_paths.append(f.name.replace(replace_sign, "/"))
    return model_paths


def check_mlops_version():
    packages = json.loads(hou.ui.packageInfo())
    variables = []
    installed_json = None
    for package in packages:
        package = packages[package]
        if "Environment variables" in package.keys():
            vars = package["Environment variables"]
            if "MLOPS" in vars.keys():
                installed_json = os.path.normpath(package["File path"])
                variables = vars.keys()
                break
    variables = list(variables)
    variables.remove("HOUDINI_PATH")

    plugin_json = os.path.normpath(hou.text.expandString("$MLOPS/MLOPs.json"))
    with open(plugin_json, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    version = None
    variables_desired = []
    for entry in data["env"]:
        variables_desired.extend(entry.keys())
        if "MLOPS_VERSION" in entry.keys():
            version = str(entry["MLOPS_VERSION"])

    message = f"Please edit:\n{installed_json}\n\nTo match the contents of:\n{plugin_json}\n\nEnvironment variables may have changed and are required for MLOPs to work properly."
    missing = [x for x in variables_desired if x not in variables]
    if len(missing) > 0:
        raise hou.Error(message + f"\nMissing Variables: {' '.join(missing)}")
    if version != hou.text.expandString("$MLOPS_VERSION"):
        raise hou.Error(
            message
            + f"\nVersion Mismatch. Configured MLOPs environment variables likely dont match those found in {plugin_json}"
        )


def ensure_huggingface_model_local(
    model_name, model_path, cache_only=False, model_type="stablediffusion"
):
    from diffusers import StableDiffusionPipeline
    from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
    from diffusers.utils import CONFIG_NAME, ONNX_WEIGHTS_NAME, WEIGHTS_NAME
    from huggingface_hub import snapshot_download

    path = hou.text.expandString(
        os.path.join(model_path, model_name.replace("/", "-_-"))
    )

    if os.path.isdir(model_name):
        return model_name
    if cache_only:
        path = os.path.normpath(path)
        if os.path.isdir(path):
            return path
        else:
            # text = "The specified model does not exist locally. Because the 'Local Cache' checkbox for this node is enabled, it also wont be downloaded automatically. Would you like to try downloading the model now?"
            # value = hou.ui.displayConfirmation(text, severity=hou.severityType.Message, title="MLOPs Missing Model")
            # if not value:
            raise hou.NodeError(
                "The specified model does not exist locally. Because the 'Local Cache' checkbox for this node is enabled, it also wont be downloaded automatically. Specify a valid local model or disable the 'Local Cache' parameter"
            )
            # cache_only = False

    model_name = model_name.replace("-_-", "/")
    allow_patterns = ["*.json", "*.txt"]
    ignore_patterns = ["*.msgpack", "*.safetensors", "*.ckpt"]

    if model_type == "stablediffusion":
        try:
            config_dict = StableDiffusionPipeline.load_config(
                model_name, resume_download=True, force_download=False
            )
            folder_names = [k for k in config_dict.keys() if not k.startswith("_")]
            allow_patterns += [os.path.join(k, "*") for k in folder_names]
            allow_patterns.append(StableDiffusionPipeline.config_name)
        except:
            pass
        allow_patterns += [
            WEIGHTS_NAME,
            SCHEDULER_CONFIG_NAME,
            CONFIG_NAME,
            ONNX_WEIGHTS_NAME,
            "*.json",
        ]
    if model_type == "transformers":
        allow_patterns.append("*.bin")
    if model_type == "all":
        allow_patterns.append("*")
        ignore_patterns = []

    snapshot_download(
        repo_id=model_name,
        local_dir=path,
        local_dir_use_symlinks=True,
        local_files_only=cache_only,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        resume_download=True,
    )
    return path.replace("\\", "/")


def pip_install(dependencies, dep_is_txt=False, upgrade=False, verbose=False):
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    cmd = ["hython", "-m", "pip", "install", "--target", PIP_FOLDER]

    if upgrade:
        cmd.append("--upgrade")

    if not dep_is_txt:
        cmd.extend(dependencies)
    else:
        cmd.append("-r")
        cmd.append(dependencies)

    env = os.environ.copy()
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]

    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=flags,
    )

    res = p.communicate()
    if res[1] != "" and p.returncode != 0:
        raise hou.Error(res[1].decode())


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
        hou.ui.displayMessage(
            f"You have successfully downloaded or updated the {model_name} model!",
            buttons=("OK",),
            severity=hou.severityType.Message,
            title="MLOPs Plugin",
        )

        self.close()

    def on_cancel(self):
        self.close()

    def open_directory_dialog(self):
        directory = hou.ui.selectFile(
            title="MLOPs - Select Download Directory",
            file_type=hou.fileType.Directory,
            multiple_select=False,
            chooser_mode=hou.fileChooserMode.Read,
        )
        if directory:
            self.download_directory_field.setText(directory)

    def buildUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle("MLOPs - Model Download")
        message_widget = QtWidgets.QLabel(
            "Automatically download a checkpoint by name from the Huggingface Hub!"
        )
        layout.addWidget(message_widget)

        layout_model = QtWidgets.QHBoxLayout()
        model_path_label = QtWidgets.QLabel("Model Name: ")
        self.model_path_field = QtWidgets.QLineEdit("runwayml/stable-diffusion-v1-5")
        layout_model.addWidget(model_path_label)
        layout_model.addWidget(self.model_path_field)

        layout_browse = QtWidgets.QHBoxLayout()
        download_path_label = QtWidgets.QLabel("Download Directory: ")
        self.download_directory_field = QtWidgets.QLineEdit("$MLOPS_MODELS/diffusers/")
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
        download_dir = hou.text.expandString(self.download_directory_field.text())

        with hou.InterruptableOperation(
            "Converting Model", open_interrupt_dialog=True
        ) as operation:
            convert_model.convert(
                checkpoint_file,
                config_file,
                download_dir,
            )

        hou.ui.displayMessage(
            f"You have successfully converted the {model_name} model!",
            buttons=("OK",),
            severity=hou.severityType.Message,
            title="MLOPs Plugin",
        )

        self.close()

    def on_cancel(self):
        self.close()

    def open_checkpoint_dialog(self):
        directory = hou.ui.selectFile(
            title="MLOPs - Model Weight",
            file_type=hou.fileType.Any,
            multiple_select=False,
            pattern="*.safetensors *.ckpt",
            chooser_mode=hou.fileChooserMode.Read,
        )
        if directory:
            self.checkpoint_file_field.setText(directory)

    def open_config_dialog(self):
        directory = hou.ui.selectFile(
            title="MLOPs - Select Config File",
            file_type=hou.fileType.Any,
            multiple_select=False,
            pattern="*.yaml",
            chooser_mode=hou.fileChooserMode.Read,
        )
        if directory:
            self.config_file_field.setText(directory)

    def open_directory_dialog(self):
        directory = hou.ui.selectFile(
            title="MLOPs - Select Download Directory",
            file_type=hou.fileType.Directory,
            multiple_select=False,
            chooser_mode=hou.fileChooserMode.Read,
        )
        if directory:
            self.download_directory_field.setText(directory)

    def buildUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle("MLOPs - Convert Model")
        message_widget = QtWidgets.QLabel(
            "Convert a checkpoint file to the Diffusers model used by this plugin!"
        )
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

        layout_browse = QtWidgets.QHBoxLayout()
        download_path_label = QtWidgets.QLabel("Download Directory: ")
        self.download_directory_field = QtWidgets.QLineEdit("$MLOPS_MODELS/diffusers/")
        layout_browse.addWidget(download_path_label)
        layout_browse.addWidget(self.download_directory_field)

        # Create QPushButton for directory browser
        self.dir_button = QtWidgets.QPushButton("...")
        self.dir_button.clicked.connect(self.open_directory_dialog)
        layout_browse.addWidget(self.dir_button)
        layout.addLayout(layout_browse)

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

        with hou.InterruptableOperation(
            "Pip Install", "Installing Dependencies", open_interrupt_dialog=True
        ) as operation:
            operation.updateLongProgress(
                percentage=-1.0, long_op_status=f"Installing dependencies"
            )
            pip_install(result, upgrade=True, verbose=True)

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
            "Enter a comma-separated list of dependencies to install"
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
