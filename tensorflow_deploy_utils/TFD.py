from datetime import datetime
import hashlib
import json
import logging
import os
import pandas as pd
from pathlib import Path
import re
import requests
import shutil
import sys
import tarfile
import tensorflow as tf
import tensorflow_text  # required if you want to load TF model using sentence piece like universal sentence encoder
from time import time


class TFD:
    def __init__(
        self,
        team: str,
        project: str,
        host: str,
        name: str = "",
        label: str = "",
        port: int = 9500,
        verbose: bool = False,
        check_connection: bool = True,
        **kwargs,
    ) -> None:
        """
        Class allow to create cursor for easy and convenient communication with TensorFlow Deploy service
        :param team: Your TEAM
        :param project: Your PROJECT
        :param host: TensorFlow Deploy service address
        :param name: (optional) Model/module NAME
        :param label: (optional) Model label (default: canary)
        :param port: (optional) TensorFlow Deploy service port (default: 9500)
        :param verbose: (optional) Verbosity (default: False)
        :param check_connection: (optional) Check connection with TensorFlow Deploy? (default: True)
        :param kwargs: optional arguments used in some methods
        """
        if verbose:
            logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        else:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        loger = logging.getLogger("TFD")

        self.team = team
        self.project = project
        self.name = name
        self.label = label
        self.host = host
        self.port = port
        self.verbose = verbose
        self.loger = loger
        self.check_connection = check_connection

        self.loger.debug(
            f"Initial params: TEAM {team}, PROJECT: {project}, NAME: {name}, LABEL: {label}, HOST: {host}, port: {port}, verbose: {verbose}"
        )

        self._check_and_prepare_params()
        if self.check_connection:
            self._check_connection()

        self.loger.info("Initialisation success!")

    def _check_and_prepare_params(self) -> None:
        """
        Internal method. It check and prepare attributes for all methods.
        :return: None
        """

        check_str_param = re.compile("[a-zA-Z0-9_]{,32}", flags=re.IGNORECASE)

        if not check_str_param.fullmatch(self.team):
            raise ValueError(f"Parameter TEAM has invalid format: {self.team}!")
        self.team = self.team.lower()

        if not check_str_param.fullmatch(self.project):
            raise ValueError(f"Parameter PROJECT has invalid format: {self.project}!")
        self.project = self.project.lower()

        if not check_str_param.fullmatch(self.name):
            raise ValueError(f"Parameter NAME has invalid format: {self.name}!")
        self.name = self.name.lower()

        if not self.label:
            self.label = "canary"
        if not check_str_param.fullmatch(self.label):
            raise ValueError(f"Parameter LABEL has invalid format: {self.label}!")
        self.label = self.label.lower()

    def _check_connection(self):
        try:
            response = requests.get(f"http://{self.host}:{self.port}/ping")
            self.loger.debug(
                f"Successful connection to TensorFlow Deploy: {response.text}"
            )
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ConnectTimeout,
        ):
            raise ConnectionError(
                f"TensorFlow Deploy on http://{self.host}:{self.port} address is NOT available"
            )

    @staticmethod
    def _calculate_hash(path: str) -> str:
        """
        Internal method. Calculate hash SHA256 for given file.
        :param path: Path to given file
        :return: String with calculated hash
        """
        sha256_hash = hashlib.sha256()

        with open(path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
            archive_hash = sha256_hash.hexdigest()

        return archive_hash

    def _action_confirmation(self, action: str, what: str, version: int = 0) -> bool:
        """
        Internal method. Ask user about action confirmation for sensitive actions.
        :param action: Action type: remove or set stable
        :param what: Action object: label, model or module
        :param version: Model or module version (optional)
        :return: True or False answer
        """
        print(f"Are You sure to {action} {what} for given parameters?")
        params = f"Parameters:\nTEAM: {self.team}\nPROJECT: {self.project}\nNAME: {self.name}\n"
        if version:
            params += f"VERSION: {version}"
        print(params)
        answer = input("Proceed? (y/n)").lower().strip()
        if answer == "y" or answer == "yes":
            return True
        else:
            return False

    def _validate_model_or_module(self, path: str) -> None:
        """
        Internal method used for validation model/module before upload.
        :param path: Full path to your TF model or module
        :return: None
        """
        self.loger.debug("model validation")
        try:
            _ = tf.saved_model.load(path)
        except Exception as error:
            raise ValueError(f"TensorFlow model validation failed! Error: {error}")
        if "README.md" not in os.listdir(path):
            raise ValueError("Directory without README.md file!")
        self.loger.debug("model validation PASS")

    def _validate_archived_model_or_module(self, path: str) -> None:
        """
        Internal method used for validation model/module archive before upload.
        :param path: Full path to your TF model or module archive
        :return: None
        """
        self.loger.debug("validation model archive")
        tmp_path = f"tmp_dir_{int(time())}"
        self.loger.debug(f"temporary directory for model: {tmp_path}")
        self._extract_archive(src_path=path, dst_path=tmp_path)
        try:
            self._validate_model_or_module(tmp_path)
        except ValueError as error:
            shutil.rmtree(tmp_path, ignore_errors=True)
            raise ValueError(error)
        self.loger.debug("removing temporary directory")
        shutil.rmtree(tmp_path, ignore_errors=True)
        self.loger.debug("validation model in archive PASS")

    def _extract_archive(self, src_path: str, dst_path: str) -> None:
        """
        Internal method used for validating models/modules. It extract archive to destination path.
        :param src_path: Full path to your TF model or module archive
        :param dst_path: Path where extract archive
        :return: None
        """
        self.loger.debug("extracting archive")
        archive = tarfile.open(src_path, "r")
        archive.extractall(dst_path)
        archive.close()
        self.loger.debug("extraction DONE")

    def create_archive(self, src_path: str, dst_path: str) -> str:
        """
        Method create tar archive with TF model or module files compatible with TensorFlow Deploy.
        :param src_path: Full path to your TF model or module
        :param dst_path: Full path to your TF model or module
        :return: String with calculated hash of archive
        """

        def tar_filter(tarinfo):
            """
            Filter takes least nested direction.
            """
            tarinfo.name = (
                "."
                + tarinfo.name[
                    len(os.path.splitdrive(str(src_path))[1].lstrip(os.path.sep)) :
                ]
            )
            return tarinfo

        # fix for path like my/path/ (ended with slash) - it causes archives with hidden files, ei. started with dot
        sep = str(os.path.sep)
        src_path = src_path.rstrip(sep)
        self.loger.debug("creating tar archive")
        archive = tarfile.open(dst_path, "w")

        archive.add(src_path, filter=tar_filter)
        archive.close()
        archive_hash = self._calculate_hash(dst_path)
        self.loger.debug("archive created")

        return archive_hash

    def delete_label(self, label: str) -> str:
        """
        Method delete given label for given model, except label: 'stable'.
        :param label: Label name
        :return: Action result
        """
        request_url = f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/names/{self.name}/labels/{label}"

        remove_decision = self._action_confirmation(action="remove", what="label")

        if not remove_decision:
            return "Nothing to do"

        response = requests.delete(request_url)
        if response.status_code != 200:
            return f"delete_label error: {response.text}"

        return f"delete_label success: {response.text}"

    def delete_model(self, version: int = None, label: str = "") -> str:
        """
        Function delete specific model from TensorFlow Deploy. Only one param of two should be given.
        :param version: Model version
        :param label: Model label
        :return: Action result
        """

        if (version and label) or not (version or label):
            raise ValueError("One of two parameters must be given: version or label")
        if label:
            request_url = f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/names/{self.name}/labels/{label}/remove_version"
        elif version:
            request_url = f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/names/{self.name}/versions/{version}"

        remove_decision = self._action_confirmation(action="remove", what="model")

        if not remove_decision:
            return "Nothing to do"

        response = requests.delete(request_url)
        if response.status_code != 200:
            return f"delete_model error: {response.text}"

        return f"delete_model success: {response.text}"

    def delete_module(self, version: int) -> str:
        """
        Method delete specific module from TensorFlow Deploy.
        :param version: Module version
        :return: Action result
        """

        request_url = f"http://{self.host}:{self.port}/v1/modules/{self.team}/{self.project}/names/{self.name}/versions/{version}"
        remove_decision = self._action_confirmation(action="remove", what="module")

        if not remove_decision:
            return "Nothing to do"

        response = requests.delete(request_url)

        if response.status_code != 200:
            return f"delete_module error: {response.text}"

        return f"delete_module success: {response.text}"

    def deploy_model(self, src_path: str, label: str = "") -> str:
        """
        Method deploy given model to production, i.e., upload model and reload all related TFS instances.
        :param src_path: Full path to model
        :param label: Label for deploying model (if give it overwrite label parameter in cursor)
        :return: Action result
        """

        upload_response = self.upload_model(src_path, label)
        if upload_response != "Upload success!":
            return f"Deploy failed. Upload error: {upload_response}"

        reload_response = self.reload_config()

        return f"Deploy results:\nupload: {upload_response}\nreload: {reload_response}"

    def generate_model_readme(
        self, dst_path: str, description: str, metrics: dict = {}
    ) -> None:
        """
        This method generate README.md file describing model and write it into given path.
        :param dst_path: Path to model
        :param description: Your additional model description
        :param metrics: Model metrics (optional)
        :return: None
        """
        readme = f"""
        # Model description
        {description}

        ## Params
        * `team`={self.team}
        * `project`={self.project}
        * `name`={self.name}
        * `label`={self.label}
        * `create_datetime`={datetime.now()}

        """

        if metrics:
            readme += "## Wyniki\n"
            for k, v in metrics.items():
                readme += f"* `{k}`={v}\n"

        with open(os.path.join(dst_path, "README.md"), "w") as fh:
            fh.write(readme)

    def generate_module_readme(self, dst_path: str, description: str) -> None:
        """
        This method generate README.md file describing module and write it into given path.
        :param dst_path: Path to module
        :param description: Your additional module description
        :return: None
        """
        readme = f"""
        # Module description
        {description}

        ## Params
        * `team`={self.team}
        * `project`={self.project}
        * `name`={self.name}
        * `create_datetime`={datetime.now()}
        """

        with open(os.path.join(dst_path, "README.md"), "w") as fh:
            fh.write(readme)

    def get_config(self) -> str:
        """
        This metod return string with model_config_file - current TFS configuration.
        :return: String with model_config_file
        """

        request_url = f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/config"
        response = requests.get(request_url)
        if response.status_code != 200:
            return f"get_config error: {response.text}"
        else:
            return response.text

    def get_model(self, dst_path: str, version: int = 0, label: str = "") -> str:
        """
        Method download specific model to given path.
        :param dst_path: Directory where write model
        :param version: Model version (priority over label, optional)
        :param label: Model label (optional)
        :return: Action result
        """
        path = Path(dst_path)
        if not path.is_dir():
            return "ERROR: dst_path is not dir"
        path = path.joinpath(f"model_{int(time())}.tar")

        if version:
            request_url = f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/names/{self.name}/versions/{version}"
        elif label:
            request_url = f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/names/{self.name}/labels/{label}"
        else:
            request_url = f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/names/{self.name}/labels/{self.label}"

        response = requests.get(request_url)
        if response.status_code != 200:
            return f"Connection error: {response.text}"

        path.write_bytes(response.content)
        return f"Model successfully written to {str(path)}"

    def get_module(self, dst_path: str, version: int) -> str:
        """
        Method download specyfic module to given path.
        :param dst_path: Directory where write module
        :param version: Module version
        :return: Action result
        """

        path = Path(dst_path)
        if not path.is_dir():
            return "ERROR: dst_path is not dir"
        path = path.joinpath(f"module_{int(time())}.tar")

        request_url = f"http://{self.host}:{self.port}/v1/modules/{self.team}/{self.project}/names/{self.name}/versions/{version}"

        response = requests.get(request_url)
        if response.status_code != 200:
            return f"Connection error: {response.text}"

        path.write_bytes(response.content)
        return f"Module successfully written to {str(path)}"

    def list_models(
        self,
        team: str = "",
        project: str = "",
        name: str = "",
        version: int = 0,
        label: str = "",
    ) -> str:
        """
        Method list models for given criteria and return them as pandas.DataFrame. If there is no search criteria,
        all models available in TensorFlow Deploy will be returned. Searching for other teams, project, etc.
        then TFD init params is also possible.
        :param team: TEAM (optional)
        :param project: PROJECT (optional)
        :param name:  NAME (optional)
        :param version: VERSION (optional)
        :param label: LABEL (optional)
        :return: pandas.DataFrame with search results
        """
        request_url = f"http://{self.host}:{self.port}/v1/models/list"
        response = requests.get(
            request_url,
            params={
                "team": team,
                "project": project,
                "name": name,
                "version": version,
                "label": label,
            },
        )
        if response.status_code != 200:
            return f"list_models error: {response.text}"

        df = pd.DataFrame(json.loads(response.text))
        if df.empty:
            return "Empty list - nothing to show"
        df.created = pd.to_datetime(df.created, unit="s")
        df.updated = pd.to_datetime(df.updated, unit="s")
        try:
            df.label = df.label.fillna("")
        except AttributeError:
            df["label"] = ""
            pass

        return df

    def list_modules(
        self, team: str = "", project: str = "", name: str = "", version: int = 0
    ) -> str:
        """
        Method list modules for given criteria and return them as pandas.DataFrame. If there is no search criteria,
        all modules available in TensorFlow Deploy will be returned. Searching for other teams, project, etc.
        then TFD init params is also possible.
        :param team: TEAM (optional)
        :param project: PROJECT (optional)
        :param name: NAME (optional)
        :param version: VERSION (optional)
        :return: pandas.DataFrame with search results
        """

        request_url = f"http://{self.host}:{self.port}/v1/modules/list"
        response = requests.get(
            request_url,
            params={"team": team, "project": project, "name": name, "version": version},
        )
        if response.status_code != 200:
            return f"list_modules error: {response.text}"

        df = pd.DataFrame(json.loads(response.text))
        if df.empty:
            return "Empty list - nothing to show"
        df.created = pd.to_datetime(df.created, unit="s")
        df.updated = pd.to_datetime(df.updated, unit="s")

        return df

    def reload_config(self, short_reload: bool = True) -> str:
        """
        Method allows you reload all TFS instances.
        :param short_reload: bool (optional): True for simple reload, False for full reload
        :return: Action result
        """
        # TODO: Need to add the `reload_status` method and modify `reload_config`
        request_url = f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/reload"
        response = requests.post(request_url, params={"SkipShortConfig": short_reload})
        if response.status_code != 200:
            return f"reload_config error: {response.text}"
        else:
            return "reload_config success!"

    def revert_model(self) -> str:
        """
        Method revert previous model stable version. It can be used only ones, because remember just last stable
        version.
        :return: Action result
        """
        request_url = f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/names/{self.name}/revert"
        response = requests.put(request_url)
        if response.status_code != 200:
            return f"revert_model error: {response.text}"
        else:
            return f"revert_model success!\n{response.text}"

    def set_label(self, version: int = None, label: str = "") -> str:
        """
        Method set given label to specific model.
        :param version: Model version
        :param label: Any model label, except: 'stable'
        :return: Action result
        """

        if not version:
            raise ValueError("You need to specify version as the first argument")

        if label:
            request_url = f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/names/{self.name}/versions/{version}/labels/{label}"
        else:
            request_url = f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/names/{self.name}/versions/{version}/labels/{self.label}"
        response = requests.put(request_url)
        if response.status_code != 200:
            return f"set_label error: {response.text}"

        reload_response = self.reload_config(short_reload=False)

        return f"set_label success: {response.text}, reload: {reload_response}"

    def set_stable(self, version: int = None, attempts: int = 3) -> str:
        """
        Method set label 'stable' to specific model. Robustness of this function is critical, hence parameter
        `attemtps` was added, to ensure stability of production environment.
        :param version: Model version
        :param attempts: Number of attempts to set label 'stable' to specific model
        :return: Action result
        """

        if not version:
            raise ValueError("You need to specify model version")

        request_url = f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/names/{self.name}/versions/{version}/labels/stable"

        action_decision = self._action_confirmation(
            action="set stable", what="label for model", version=version
        )

        if not action_decision:
            return "Nothing to do"

        errors = []
        for i in range(attempts):
            response = requests.put(request_url)
            if response.status_code != 200:
                errors.append(f"#{i} error: {response.text}")
            else:
                break
        else:
            return f"set_stable error! Errors from all attempts: {errors}"

        reload_response = self.reload_config(short_reload=False)
        return f"set_stable success: {response.text}, reload status: {reload_response}"

    def upload_model(self, src_path: str, label: str = "", timeout: int = 120) -> str:
        """
        Method upload directory/archive containing TF model to TensorFlow Deploy.
        :param src_path: Full path to model. It can also be already archived model
        :param label: (Optional) Label for model
        :param timeout: Upload timeout
        :return: Action result
        """
        path = Path(src_path)
        if not label:
            label = self.label
        self.loger.debug(f"src_path: {src_path}")
        if path.is_dir():
            self.loger.debug("src_path is a directory")
            self._validate_model_or_module(src_path)
            dst_path = Path(f"tmp_upload_{int(time())}.tar")
            archive_hash = self.create_archive(src_path=src_path, dst_path=dst_path)
            self.loger.debug(f"archive path: {dst_path}")
            self.loger.debug(f"archive hash: {archive_hash}")
            f = open(dst_path, "rb")
            multipart_form_data = {
                "archive_data": (dst_path.name, f),
                "archive_hash": archive_hash,
            }
            self.loger.debug("uploading archive")
            response = requests.post(
                f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/names/{self.name}/labels/{label}",
                files=multipart_form_data,
                timeout=timeout,
            )
            f.close()
            dst_path.unlink()
            self.loger.debug(f"upload result: {response.text}")
            if response.status_code != 200:
                return f"Upload failed!\nServer response: {response.text}"

            return "Upload success!"
        elif path.suffix != ".tar":
            self.loger.debug("src_path in not a tar archive")
            return "Unexpected file extension. src_path must be a tar archive"
        else:
            self.loger.debug("src_path is tar archive")
            self._validate_archived_model_or_module(str(path))
            path.open()
            self.loger.debug("calculating hash")
            archive_hash = self._calculate_hash(str(path))
            self.loger.debug(f"archive hash: {archive_hash}")
            f = open(str(path), "rb")
            multipart_form_data = {
                "archive_data": (path.name, f),
                "archive_hash": archive_hash,
            }
            self.loger.debug("uploading archive")
            response = requests.post(
                f"http://{self.host}:{self.port}/v1/models/{self.team}/{self.project}/names/{self.name}/labels/{label}",
                files=multipart_form_data,
                timeout=timeout,
            )
            f.close()
            os.remove(str(path))
            self.loger.debug(f"upload result: {response.text}")
            if response.status_code != 200:
                return f"Upload failed!\nServer response: {response.text}"
            return "Upload success!"

    def upload_module(self, src_path: str, timeout: int = 600):
        """
        Method upload directory/archive containing TF module to TensorFlow Deploy.
        :param src_path: Full path to module. It can also be already archived module
        :param timeout: Upload timeout in seconds
        :return: Action result
        """

        path = Path(src_path)
        self.loger.debug(f"src_path: {src_path}")
        if path.is_dir():
            self.loger.debug("src_path is a directory")
            self._validate_model_or_module(src_path)
            dst_path = Path(f"tmp_upload_{int(time())}.tar")
            archive_hash = self.create_archive(src_path=src_path, dst_path=dst_path)
            f = open(dst_path, "rb")
            multipart_form_data = {
                "archive_data": (dst_path.name, f),
                "archive_hash": archive_hash,
            }
            self.loger.debug("uploading archive")
            response = requests.post(
                f"http://{self.host}:{self.port}/v1/modules/{self.team}/{self.project}/names/{self.name}",
                files=multipart_form_data,
                timeout=timeout,
            )
            f.close()
            dst_path.unlink()
            self.loger.debug(f"upload result: {response.text}")
            if response.status_code != 200:
                return f"Upload failed!\nServer response: {response.text}"
            return "Upload success!"
        elif path.suffix != ".tar":
            self.loger.debug("src_path in not a tar archive")
            return "Unexpected file extension. src_path must be a tar archive"
        else:
            self.loger.debug("src_path is tar archive")
            self._validate_archived_model_or_module(str(path))
            path.open()
            self.loger.debug("calculating hash")
            archive_hash = self._calculate_hash(str(path))
            f = open(str(path), "rb")
            multipart_form_data = {
                "archive_data": (path.name, f),
                "archive_hash": archive_hash,
            }
            self.loger.debug("uploading archive")
            response = requests.post(
                f"http://{self.host}:{self.port}/v1/modules/{self.team}/{self.project}/names/{self.name}",
                files=multipart_form_data,
                timeout=timeout,
            )
            f.close()
            os.remove(str(path))
            self.loger.debug(f"upload result: {response.text}")
            if response.status_code != 200:
                return f"Upload failed!\nServer response: {response.text}"
            return "Upload success!"

    def __str__(self):
        return f"TensorFlow Deploy cursor\nTEAM: {self.team}\nPROJECT: {self.project}\nNAME: {self.name}\nLABEL: {self.label}\nHost: {self.host}\nPort: {self.port}\nVerbose: {self.verbose}\nCheck connection: {self.check_connection}"
