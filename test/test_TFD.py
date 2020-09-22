#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import logging
import tarfile
import time
import unittest
import unittest.mock as mock

import pandas as pd
import requests_mock

from tensorflow_deploy_utils import TFD

# Disable TFD logger messages
logging.disable(logging.CRITICAL)

# TFD Api Url Adressess
endpoint = {
    "ping": "http://{host}:{port}/ping",
    "label": "http://{host}:{port}/v1/models/{team}/{project}/names/{name}/labels/{label}",
    "model_v": "http://{host}:{port}/v1/models/{team}/{project}/names/{name}/versions/{version}",
    "model_l": "http://{host}:{port}/v1/models/{team}/{project}/names/{name}/labels/{label}/remove_version",
    "modules_v": "http://{host}:{port}/v1/modules/{team}/{project}/names/{name}/versions/{version}",
    "modules": "http://{host}:{port}/v1/modules/{team}/{project}/names/{name}",
    "list_models": "http://{host}:{port}/v1/models/list",
    "list_modules": "http://{host}:{port}/v1/modules/list",
    "config": "http://{host}:{port}/v1/models/{team}/{project}/config",
    "reload": "http://{host}:{port}/v1/models/{team}/{project}/reload?SkipShortConfig={reload_type}",
    "set_status": "http://{host}:{port}/v1/models/{team}/{project}/names/{name}/versions/{version}/labels/stable",
    "set_label": "http://{host}:{port}/v1/models/{team}/{project}/names/{name}/versions/{version}/labels/{label}",
    "revert_model": "http://{host}:{port}/v1/models/{team}/{project}/names/{name}/revert",
}


class TestTFD(unittest.TestCase):

    # variables
    host = "test_host"
    team = "test_team"
    project = "test_project"
    name = "test_name"
    label = "test_label"
    port = 9500
    version = 1
    check_connection = False
    verbose = False

    @requests_mock.mock()
    def setUp(self, requests_mock):
        """
        Sets the TFD cursor for testing.
        """

        url = endpoint["ping"].format(host=self.host, port=self.port)
        requests_mock.get(
            url,
            status_code=200,
        )
        self.tfd_cursor = TFD(
            host=self.host,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
            check_connection=self.check_connection,
        )

    def test_str(self):
        """
        Checks if __str__ returns valid value.
        """

        return__str__test = self.tfd_cursor.__str__()
        expected = "TensorFlow Deploy cursor\nTEAM: {t}\nPROJECT: {p}\nNAME: {n}\nLABEL: {l}\nHost: {h}\nPort: {po}\n" "Verbose: {v}\nCheck connection: {c}".format(
            t=self.team,
            p=self.project,
            n=self.name,
            l=self.label,
            h=self.host,
            po=self.port,
            v=self.verbose,
            c=self.check_connection,
        )
        self.assertEqual(
            return__str__test,
            expected,
            msg="Got '{t}' expected '{e}'".format(t=return__str__test, e=expected),
        )

    @requests_mock.mock()
    def test_TFDclass(self, requests_mock):
        """
        Scenario checks that cursor can be set with verbose on or off and connection check on or off.
        In any case, the operation should be successful.
        """

        url = endpoint["ping"].format(host=self.host, port=self.port)
        requests_mock.get(
            url,
            status_code=200,
        )
        tfd_cursor = TFD(
            verbose=True,
            host=self.host,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )
        self.assertIsInstance(tfd_cursor, object)
        tfd_cursor = TFD(
            verbose=False,
            host=self.host,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )
        self.assertIsInstance(tfd_cursor, object)
        tfd_cursor = TFD(
            check_connection=True,
            host=self.host,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )
        self.assertIsInstance(tfd_cursor, object)
        tfd_cursor = TFD(
            check_connection=False,
            host=self.host,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )
        self.assertIsInstance(tfd_cursor, object)

    def test_check_params_err(self):
        """
        Scenario checks if params validator raises exception ValueError on bad values.
        The function should return an appropriate error.
        """

        raised = None

        # test parameter values
        cases = [
            "test!",
            "t@est",
            "t#st",
            "t{est",
            "test$",
            "testtest^",
            "te st",
            "test_test&",
            "(test",
            ")test",
            "te\\st",
            ".test",
            "asgdgfgfhrhthfhghshghghghghghge34",  # longer than 32 chars
            "汉字漢字",
            "БГПжы",
        ]
        # test parameter scenarios values
        scenarios = ["TEAM", "PROJECT", "NAME", "LABEL"]

        for scenario in scenarios:
            for test_value in cases:
                try:
                    if scenario == "TEAM":
                        _ = TFD(
                            check_connection=False,
                            host=self.host,
                            team=test_value,
                            project=self.project,
                            name=self.name,
                            label=self.label,
                        )
                    elif scenario == "PROJECT":
                        _ = TFD(
                            check_connection=False,
                            host=self.host,
                            team=self.team,
                            project=test_value,
                            name=self.name,
                            label=self.label,
                        )
                    elif scenario == "NAME":
                        _ = TFD(
                            check_connection=False,
                            host=self.host,
                            team=self.team,
                            project=self.project,
                            name=test_value,
                            label=self.label,
                        )
                    elif scenario == "LABEL":
                        _ = TFD(
                            check_connection=False,
                            host=self.host,
                            team=self.team,
                            project=self.project,
                            name=self.name,
                            label=test_value,
                        )
                except ValueError as exceptionMessage:
                    raised = exceptionMessage
                expected = "Parameter {scenario} has invalid format: {invalid_parameter}!".format(
                    invalid_parameter=test_value, scenario=scenario
                )
                self.assertEqual(expected, str(raised))
                self.assertIsNotNone(
                    raised, "Should get exception, not: {e}".format(e=raised)
                )

    def test_TFDclass_noparams(self):
        """
        Scenario checks that cursor can be set with empty label.
        Should return object with param label as canary.
        """

        tfd_cursor = TFD(
            check_connection=False,
            host=self.host,
            team=self.team,
            project=self.project,
            name=self.name,
        )
        self.assertEqual(tfd_cursor.label, "canary")

    @requests_mock.mock()
    def test_check_connection(self, requests_mock):
        """
        Scenario check connection_check function.
        On success, the function should not return anything.
        """

        url = endpoint["ping"].format(host=self.host, port=self.port)
        requests_mock.get(
            url,
            status_code=200,
        )
        raised = False
        err_msg = None
        try:
            self.tfd_cursor._check_connection()
        except ConnectionError as error_message:
            raised = True
            err_msg = error_message
        self.assertFalse(raised, "Should not get exception: {e}".format(e=err_msg))

    def test_check_connection_err(self):
        """
        Scenario tests if check connection raises exception ConnectionError when can't reach the TFD Api.
        If the connection fails, should return error.
        """

        conn_error = None
        try:
            self.tfd_cursor._check_connection()
        except ConnectionError as error_message:
            conn_error = error_message
        expected = (
            "TensorFlow Deploy on http://{host}:{port} address is NOT available".format(
                host=self.host, port=self.port
            )
        )
        self.assertIsNotNone(
            conn_error, msg="Expected None, got {r}".format(r=conn_error)
        )
        self.assertEqual(
            expected,
            str(conn_error),
            msg="Expected: '{error}', got '{response}'".format(
                error=expected, response=str(conn_error)
            ),
        )

    @requests_mock.mock()
    @mock.patch("builtins.input", return_value="y")
    def test_delete_label_useryes(self, requests_mock, input_mock):
        """
        Scenario checks if function delete_label responded correctly when succeeded.
        """

        url = endpoint["label"].format(
            name=self.name,
            label=self.label,
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
        )
        requests_mock.delete(
            url,
            text="label deleted",
            status_code=200,
        )
        expected = "delete_label success: label deleted"
        response = self.tfd_cursor.delete_label(self.label)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @mock.patch("builtins.input", return_value="n")
    def test_delete_label_userno(self, input_mock):
        """
        Scenario checks delete_label function when user input is "no" in action confirmation.
        The function should be interrupted with "Nothing to do" message.
        """

        expected = "Nothing to do"
        response = self.tfd_cursor.delete_label(self.label)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    @mock.patch("builtins.input", return_value="y")
    def test_delete_label_useryes_err(self, requests_mock, input_mock):
        """
        Scenario checks delete_label function when something went wrong.
        Should return error.
        """

        url = endpoint["label"].format(
            name=self.name,
            label=self.label,
            host=self.host,
            port=9500,
            team=self.team,
            project=self.project,
        )
        requests_mock.delete(
            url,
            text="error msg",
            status_code=500,
        )
        expected = "delete_label error: error msg"
        response = self.tfd_cursor.delete_label(self.label)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @mock.patch("builtins.input", return_value="y")
    def test_delete_model_noparams_err(self, input_mock):
        """
        Scenario checks delete_model function with user input "yes" at action confirmation.
        Should return valid response.
        """

        expected = "One of two parameters must be given: version or label"
        try:
            self.tfd_cursor.delete_model()
        except ValueError as err:
            error = str(err)
        self.assertEqual(
            error,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, error),
        )

    @requests_mock.mock()
    @mock.patch("builtins.input", return_value="y")
    def test_delete_model_label_useryes(self, requests_mock, input_mock):
        """
        Scenario checks delete_model function with user input "yes" at action confirmation.
        Should return valid response.
        """

        url = endpoint["model_l"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
            host=self.host,
            port=self.port,
        )

        requests_mock.delete(
            url,
            text="model deleted",
            status_code=200,
        )
        expected = "delete_model success: model deleted"
        response = self.tfd_cursor.delete_model(label=self.label)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    @mock.patch("builtins.input", return_value="y")
    def test_delete_model_version_useryes(self, requests_mock, input_mock):
        """
        Scenario checks delete_model function with user input "yes" at action confirmation.
        Should return valid response.
        """

        url = endpoint["model_v"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            host=self.host,
            port=self.port,
        )

        requests_mock.delete(
            url,
            text="model deleted",
            status_code=200,
        )
        expected = "delete_model success: model deleted"
        response = self.tfd_cursor.delete_model(self.version)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @mock.patch("builtins.input", return_value="n")
    def test_delete_model_userno(self, input_mock):
        """
        Scenario checks delete_model function when user input is "no" in confirmation action.
        The function should be interrupted and return "Nothing to do" message.
        """

        expected = "Nothing to do"
        response = self.tfd_cursor.delete_model(self.version)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    @mock.patch("builtins.input", return_value="y")
    def test_delete_model_useryes_err(self, requests_mock, input_mock):
        """
        Scenario tests the delete_model function when application responded with a code other than 200.
        Should return error.
        """

        url = endpoint["model_v"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            host=self.host,
            port=self.port,
        )

        requests_mock.delete(
            url,
            text="error msg",
            status_code=500,
        )
        expected = "delete_model error: error msg"
        response = self.tfd_cursor.delete_model(self.version)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    @mock.patch("builtins.input", return_value="y")
    def test_delete_module_useryes(self, requests_mock, input_mock):
        """
        Scenario tests delete_module function.
        Should return message about removing the module.
        """

        url = endpoint["modules_v"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            host=self.host,
            port=self.port,
        )

        requests_mock.delete(
            url,
            text="module deleted",
            status_code=200,
        )
        expected = "delete_module success: module deleted"
        response = self.tfd_cursor.delete_module(self.version)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @mock.patch("builtins.input", return_value="n")
    def test_delete_module_userno(self, input_mock, version=1):
        """
        Scenario tests delete_module function with user input "no" at action confirmation.
        The function should be interrupted with "Nothing to do" message.
        """

        expected = "Nothing to do"
        response = self.tfd_cursor.delete_module(version=self.version)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    @mock.patch("builtins.input", return_value="y")
    def test_delete_module_useryes_err(self, requests_mock, input_mock):
        """
        Scenario tests delete_module func with user input "yes" at action confirmation
        and api responded with status code other than 200.
        Function should return an error message.
        """

        url = endpoint["modules_v"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            host=self.host,
            port=self.port,
        )

        requests_mock.delete(
            url,
            text="error msg",
            status_code=500,
        )
        expected = "delete_module error: error msg"
        response = self.tfd_cursor.delete_module(self.version)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    def test_get_config(self, requests_mock):
        """
        Scenario tests get_config function.
        Should return config as result.
        """

        url = endpoint["config"].format(
            host=self.host, port=self.port, team=self.team, project=self.project
        )

        requests_mock.get(
            url,
            text="config",
            status_code=200,
        )
        expected = "config"
        response = self.tfd_cursor.get_config()
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    def test_get_config_err(self, requests_mock):
        """
        Scenario tests get_config function when response from TDF Api is not 200.
        Should return message that says config cannot be downloaded.
        """
        url = endpoint["config"].format(
            host=self.host, port=self.port, team=self.team, project=self.project
        )
        requests_mock.get(
            url,
            text="error msg",
            status_code=500,
        )
        expected = "get_config error: error msg"
        response = self.tfd_cursor.get_config()
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    def test_get_model_version(self, requests_mock):
        """
        Scenario tests get_module function with version as param.
        Should return success message.
        """

        path = "/dir/to/test/"
        r_path = path + "model_{t}.tar".format(t=int(time.time()))

        version_url = endpoint["model_v"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
        )

        requests_mock.get(
            version_url,
            text="Model successfully written to {p}".format(p=path),
            status_code=200,
        )
        expected = "Model successfully written to {p}".format(p=r_path)
        with mock.patch("pathlib.Path.write_bytes", return_value="response.content"):
            with mock.patch("pathlib.Path.is_dir", return_value=True):
                response = self.tfd_cursor.get_model(path, self.version, self.label)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    def test_get_model_label(self, requests_mock):
        """
        Scenario tests get_model function with label as param.
        Should return success message.
        """

        path = "/dir/to/test/"
        r_path = path + "model_{t}.tar".format(t=int(time.time()))
        url = endpoint["label"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )

        requests_mock.get(
            url,
            text="Model successfully written to {p}".format(p=path),
            status_code=200,
        )
        expected = "Model successfully written to {p}".format(p=r_path)
        with mock.patch("pathlib.Path.write_bytes", return_value="response.content"):
            with mock.patch("pathlib.Path.is_dir", return_value=True):
                response = self.tfd_cursor.get_model(dst_path=path, label=self.label)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    def test_get_model(self, requests_mock):
        """
        Scenario tests get_model function when label or version were not given.
        Should return success message.
        """

        path = "/dir/to/test/"
        r_path = path + "model_{t}.tar".format(t=int(time.time()))
        url = endpoint["label"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )

        requests_mock.get(
            url,
            text="Model successfully written to {p}".format(p=path),
            status_code=200,
        )
        expected = "Model successfully written to {p}".format(p=r_path)
        with mock.patch("pathlib.Path.write_bytes", return_value="response.content"):
            with mock.patch("pathlib.Path.is_dir", return_value=True):
                response = self.tfd_cursor.get_model(dst_path=path)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    def test_get_model_label_err(self, requests_mock):
        """
        Scenario tests get_model function with label as param when something went wrong with TFD Api.
        Should return error.
        """

        path = "/dir/to/test/"
        url = endpoint["label"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )
        requests_mock.get(
            url,
            text="error msg",
            status_code=500,
        )
        expected = "Connection error: error msg"
        with mock.patch("pathlib.Path.is_dir", return_value=True):
            response = self.tfd_cursor.get_model(dst_path=path, label=self.label)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    def test_get_model_version_err(self, requests_mock):
        """
        Scenario tests get_model function with version as param when something is wrong with TFD Api.
        Should return error.
        """

        path = "/dir/to/test/"
        version_url = endpoint["model_v"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
        )
        requests_mock.get(
            version_url,
            text="error msg",
            status_code=500,
        )
        expected = "Connection error: error msg"
        with mock.patch("pathlib.Path.is_dir", return_value=True):
            response = self.tfd_cursor.get_model(path, self.version, self.label)
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    def test_get_model_version_dir_err(self, requests_mock):
        """
        Scenario tests get_model function with version as param and when bad dir is also given.
        Should return directory error.
        """

        path = "/dir/to/test/"
        response = self.tfd_cursor.get_model(path, self.version, self.label)
        expected = "ERROR: dst_path is not dir"
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    def test_get_module(self, requests_mock):
        """
        Scenario tests get_module function.
        Should return success message.
        """

        path = "/dir/to/test/"
        r_path = path + "module_{t}.tar".format(t=int(time.time()))

        url = endpoint["modules_v"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
        )

        requests_mock.get(
            url,
            text="Module successfully written to {p}".format(p=path),
            status_code=200,
        )
        expected = "Module successfully written to {p}".format(p=r_path)
        with mock.patch("pathlib.Path.write_bytes", return_value="response.content"):
            with mock.patch("pathlib.Path.is_dir", return_value=True):
                response = self.tfd_cursor.get_module(path, self.version)

        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    def test_get_module_err(self, request_mock):
        """
        Scenario tests get_module function when something is wrong with TFD Api.
        Should return error.
        """

        path = "/dir/to/test/"
        url = endpoint["modules_v"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
        )
        request_mock.get(
            url,
            text="error msg",
            status_code=500,
        )
        with mock.patch("pathlib.Path.is_dir", return_value=True):
            response = self.tfd_cursor.get_module(path, self.version)

        expected = "Connection error: error msg"
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    def test_get_module_dir_err(self, requests_mock):
        """
        Scenario tests get_module function with bad directory given.
        Should return error.
        """

        path = "/dir/to/test/"
        response = self.tfd_cursor.get_module(path, self.version)
        expected = "ERROR: dst_path is not dir"
        self.assertEqual(
            response,
            expected,
            msg="Expected: '{}', got '{}'".format(expected, response),
        )

    @requests_mock.mock()
    def test_list_models(self, requests_mock):
        """
        Scenario tests list_models function.
        Should return DataFrame.
        """

        now = time.time()
        payload_data = {
            "team": [self.team],
            "project": [self.project],
            "version": [1],
            "label": [self.label],
            "id": [1],
            "status": ["ready"],
            "created": [now],
            "updated": [now],
        }
        payload = json.dumps(payload_data)
        url = endpoint["list_models"].format(host=self.host, port=self.port)

        requests_mock.get(
            url,
            text=payload,
            status_code=200,
        )
        response = self.tfd_cursor.list_models(
            self.team, self.project, self.name, 1, self.label
        )
        excepted = {
            "team": [self.team],
            "project": [self.project],
            "version": [1],
            "label": [self.label],
            "id": [1],
            "status": ["ready"],
            "created": [now],
            "updated": [now],
        }
        df_expected = pd.DataFrame(excepted)
        df_expected.created = pd.to_datetime(df_expected.created, unit="s")
        df_expected.updated = pd.to_datetime(df_expected.updated, unit="s")
        equals = df_expected.equals(response)

        self.assertTrue(equals)

    @requests_mock.mock()
    def test_list_models_nolabel(self, requests_mock):
        """
        Scenario tests list_models function.
        Should return DataFrame.
        """

        now = time.time()
        payload_data = {
            "team": [self.team],
            "project": [self.project],
            "version": [1],
            "id": [1],
            "status": ["ready"],
            "created": [now],
            "updated": [now],
        }
        payload = json.dumps(payload_data)
        url = endpoint["list_models"].format(host=self.host, port=self.port)

        requests_mock.get(
            url,
            text=payload,
            status_code=200,
        )
        response = self.tfd_cursor.list_models(
            self.team, self.project, self.name, 1, self.label
        )
        excepted = {
            "team": [self.team],
            "project": [self.project],
            "version": [1],
            "id": [1],
            "status": ["ready"],
            "created": [now],
            "updated": [now],
            "label": "",
        }
        df_expected = pd.DataFrame(excepted)
        df_expected.created = pd.to_datetime(df_expected.created, unit="s")
        df_expected.updated = pd.to_datetime(df_expected.updated, unit="s")
        equals = df_expected.equals(response)

        self.assertTrue(equals)

    @requests_mock.mock()
    def test_list_models_err(self, requests_mock):
        """
        Scenario tests list_models function when TFD Api not responding correctly.
        Should return error.
        """

        url = endpoint["list_models"].format(host=self.host, port=self.port)
        requests_mock.get(
            url,
            text="error",
            status_code=500,
        )
        response = self.tfd_cursor.list_models(
            self.team, self.project, self.name, 1, self.label
        )
        expected = "list_models error: error"
        self.assertEqual(
            response,
            expected,
            msg="Got '{r}', expected: '{e}'".format(r=response, e=expected),
        )

    @requests_mock.mock()
    def test_list_models_empty(self, requests_mock):
        """
        Scenario tests list_models function when empty data was given as input.
        The result should say that there is nothing to show.
        """

        payload_data = {
            "team": [],
            "project": [],
            "version": [],
            "label": [],
            "id": [],
            "status": [],
            "created": [],
            "updated": [],
        }
        payload = json.dumps(payload_data)
        url = endpoint["list_models"].format(host=self.host, port=self.port)
        requests_mock.get(
            url,
            text=payload,
            status_code=200,
        )
        response = self.tfd_cursor.list_models()
        expected = "Empty list - nothing to show"
        self.assertEqual(
            response,
            expected,
            msg="Got: {r}, expected: {e}".format(r=response, e=expected),
        )

    @requests_mock.mock()
    def test_list_modules(self, requests_mock):
        """
        Scenario tests list_modules function.
        Should return DataFrame.
        """

        now = time.time()
        payload_data = {
            "team": [self.team],
            "project": [self.project],
            "version": [self.version],
            "label": [self.label],
            "id": [1],
            "status": ["ready"],
            "created": [now],
            "updated": [now],
        }
        payload = json.dumps(payload_data)
        url = endpoint["list_modules"].format(host=self.host, port=self.port)

        requests_mock.get(
            url,
            text=payload,
            status_code=200,
        )
        response = self.tfd_cursor.list_modules(self.team, self.project, self.name, 1)
        expected = {
            "team": [self.team],
            "project": [self.project],
            "version": [self.version],
            "label": [self.label],
            "id": [1],
            "status": ["ready"],
            "created": [now],
            "updated": [now],
        }
        df_expected = pd.DataFrame(expected)
        df_expected.created = pd.to_datetime(df_expected.created, unit="s")
        df_expected.updated = pd.to_datetime(df_expected.updated, unit="s")

        equals = df_expected.equals(response)

        self.assertTrue(
            equals, msg="Got '{r}', expected '{e}'".format(r=expected, e=response)
        )

    @requests_mock.mock()
    def test_list_modules_err(self, requests_mock):
        """
        Scenario tests list modules function when something went wrong with TFD Api.
        Should return error.
        """

        url = endpoint["list_modules"].format(host=self.host, port=self.port)
        requests_mock.get(
            url,
            text="error",
            status_code=500,
        )
        response = self.tfd_cursor.list_modules(self.team, self.project, self.name, 1)
        expected = "list_modules error: error"
        self.assertEqual(
            response,
            expected,
            msg="Got '{r}', expected: '{e}'".format(r=response, e=expected),
        )

    @requests_mock.mock()
    def test_list_modules_empty(self, requests_mock):
        """
        Scenario tests list_modules function when empty data was given as input.
        The result should say that there is nothing to show.
        """

        payload_data = {
            "team": [],
            "project": [],
            "version": [],
            "label": [],
            "id": [],
            "status": [],
            "created": [],
            "updated": [],
        }
        payload = json.dumps(payload_data)
        url = endpoint["list_modules"].format(host=self.host, port=self.port)
        requests_mock.get(
            url,
            text=payload,
            status_code=200,
        )
        response = self.tfd_cursor.list_modules()
        expected = "Empty list - nothing to show"

        self.assertEqual(
            response,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=response, e=expected),
        )

    @requests_mock.mock()
    def test_reload_config(self, requests_mock):
        """
        Scenario tests reload_config function.
        Function should return information that the operation was successful.
        """

        url = endpoint["reload"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            reload_type=True,
        )
        requests_mock.post(
            url,
            text="",
            status_code=200,
        )
        expected = "reload_config success!"
        response = self.tfd_cursor.reload_config()

        self.assertEqual(
            expected,
            response,
            msg="Got '{r}', expected '{e}'".format(r=response, e=expected),
        )

    @requests_mock.mock()
    def test_reload_config_err(self, requests_mock):
        """
        Scenario tests reload_config function when something is wrong with TFD Api.
        Should return error.
        """

        url = endpoint["reload"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            reload_type=True,
        )
        requests_mock.post(
            url,
            text="error msg",
            status_code=500,
        )
        expected = "reload_config error: {r}".format(r="error msg")
        response = self.tfd_cursor.reload_config()

        self.assertEqual(
            expected,
            response,
            msg="Got '{r}', expected '{e}'".format(r=response, e=expected),
        )

    @requests_mock.mock()
    def test_revert_model(self, requests_mock):
        """
        Scenario tests revert_model function.
        Should return information that model was reverted.
        """
        url = endpoint["revert_model"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
        )
        requests_mock.put(
            url,
            text="model reverted",
            status_code=200,
        )
        expected = "revert_model success!\nmodel reverted"
        response = self.tfd_cursor.revert_model()

        self.assertEqual(
            expected,
            response,
            msg="Got '{r}', expected '{e}'".format(r=response, e=expected),
        )

    @requests_mock.mock()
    def test_revert_model_err(self, requests_mock):
        """
        Scenario tests revert_model function when something is wrong with TFD Api.
        It should return an error that the operation has not been performed.
        """

        url = endpoint["revert_model"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
        )
        requests_mock.put(
            url,
            text="error msg",
            status_code=500,
        )
        expected = "revert_model error: {r}".format(r="error msg")
        response = self.tfd_cursor.revert_model()
        self.assertEqual(
            expected,
            response,
            msg="Got '{r}', expected '{e}'".format(r=response, e=expected),
        )

    @requests_mock.mock()
    def test_set_label(self, requests_mock):
        """
        Scenario tests set_label function with label as param.
        Should return that the label has been setted.
        """

        url = endpoint["set_label"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            label=self.label,
            host=self.host,
            port=self.port,
        )

        requests_mock.put(
            url,
            text="label set",
            status_code=200,
        )

        url_reload = endpoint["reload"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            label=self.label,
            host=self.host,
            port=self.port,
            reload_type=False,
        )
        requests_mock.post(
            url_reload,
            text="reload_config success!",
            status_code=200,
        )
        expected = "set_label success: label set, reload: reload_config success!"
        response = self.tfd_cursor.set_label(version=self.version, label=self.label)
        self.assertEqual(
            expected,
            response,
            msg="Got '{r}', expected '{e}'".format(r=response, e=expected),
        )

    @requests_mock.mock()
    def test_set_label_version(self, requests_mock):
        """
        Scenario tests set_label function with version as param.
        Should return that the label has been setted.
        """

        url = endpoint["set_label"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            label=self.label,
            host=self.host,
            port=self.port,
        )

        url_reload = endpoint["reload"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            label=self.label,
            host=self.host,
            port=self.port,
            reload_type=False,
        )
        requests_mock.post(
            url_reload,
            text="reload_config success!",
            status_code=200,
        )

        requests_mock.put(
            url,
            text="label set",
            status_code=200,
        )
        response = self.tfd_cursor.set_label(self.version)
        expected = "set_label success: label set, reload: reload_config success!"
        self.assertEqual(
            expected,
            response,
            msg="Got '{r}', expected '{e}'".format(r=response, e=expected),
        )

    @requests_mock.mock()
    def test_set_label_version_err(self, requests_mock):
        """
        Scenario tests set_label function with version as param when TFD Api not responding correctly.
        Should return a message saying that the label could not be setted.
        """

        url = endpoint["set_label"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            label=self.label,
            host=self.host,
            port=self.port,
        )
        requests_mock.put(
            url,
            text="error msg",
            status_code=500,
        )
        expected = "set_label error: error msg"
        response = self.tfd_cursor.set_label(self.version)
        self.assertEqual(
            expected,
            response,
            msg="Got '{r}', expected '{e}'".format(r=response, e=expected),
        )

    def test_set_label_noversion_err(self):
        """
        Scenario tests set_label function with no version given.
        Should return ValueError.
        """
        try:
            response = self.tfd_cursor.set_label()
        except ValueError as err:
            error = str(err)

        expected = "You need to specify version as the first argument"
        self.assertEqual(
            expected,
            error,
            msg="Got '{r}', expected '{e}'".format(r=error, e=expected),
        )

    @requests_mock.mock()
    def test_set_label_err(self, requests_mock):
        """
        Scenario tests set_label function with label as param when TFD Api not responding correctly.
        Should return a message saying that the label could not be setted.
        """

        url = endpoint["set_label"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            label=self.label,
            host=self.host,
            port=self.port,
        )
        requests_mock.put(
            url,
            text="error msg",
            status_code=500,
        )
        expected = "set_label error: error msg"
        response = self.tfd_cursor.set_label(self.version, self.label)
        self.assertEqual(
            expected,
            response,
            msg="Got '{r}', expected '{e}'".format(r=response, e=expected),
        )

    @requests_mock.mock()
    @mock.patch("builtins.input", return_value="y")
    def test_set_stable(self, requests_mock, input_mock):
        """
        Scenario tests set_stable function.
        Should return that model status was changed to stable.
        """

        msg = "label set stable"

        url = endpoint["set_status"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            host=self.host,
            port=self.port,
        )
        requests_mock.put(
            url,
            text=msg,
            status_code=200,
        )

        url_reload = endpoint["reload"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            host=self.host,
            port=self.port,
            reload_type=False,
        )

        requests_mock.post(
            url_reload,
            text="reload_config success!",
            status_code=200,
        )

        result = self.tfd_cursor.set_stable(self.version)
        expected = "set_stable success: label set stable, reload status: reload_config success!"
        self.assertEqual(
            result,
            expected,
            msg="Got '{r}', expected: '{e}'".format(r=result, e=expected),
        )

    def test_set_stable_noversion_err(self):
        """
        Scenario tests set_stable function with no version given.
        Should return ValueError.
        """
        try:
            response = self.tfd_cursor.set_stable()
        except ValueError as err:
            error = str(err)

        expected = "You need to specify model version"
        self.assertEqual(
            expected,
            error,
            msg="Got '{r}', expected '{e}'".format(r=error, e=expected),
        )

    @requests_mock.mock()
    @mock.patch("builtins.input", return_value="y")
    def test_set_stable_err(self, requests_mock, input_mock):
        """
        Scenario tests set_stable function when something is wrong with TFD Api.
        Should return error.
        """

        attempts = 3
        url = endpoint["set_status"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            host=self.host,
            port=self.port,
        )
        err = "error label not set stable"
        requests_mock.put(
            url,
            text=err,
            status_code=500,
        )
        result = self.tfd_cursor.set_stable(self.version)

        errors = []
        for i in range(attempts):
            errors.append("#{i} error: {e}".format(i=i, e=err))

        expected = "set_stable error! Errors from all attempts: {e}".format(e=errors)
        self.assertEqual(
            result,
            expected,
            msg="Got '{r}', expected: '{e}'".format(r=result, e=expected),
        )

    @requests_mock.mock()
    @mock.patch("builtins.input", return_value="n")
    def test_set_stable_userno(self, requests_mock, input_mock):
        """
        Scenario tests set_stable function when user input at action confirmation was "no".
        The function should be interrupted with a message "Nothing to do".
        """

        attempts = 3
        url = endpoint["set_status"].format(
            team=self.team,
            project=self.project,
            name=self.name,
            version=self.version,
            host=self.host,
            port=self.port,
        )
        err = "error label not set stable"
        requests_mock.put(
            url,
            text=err,
            status_code=500,
        )
        result = self.tfd_cursor.set_stable(self.version)

        errors = []
        for i in range(attempts):
            errors.append("#{i} error: {e}".format(i=i, e=err))

        expected = "Nothing to do"
        self.assertEqual(
            result,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=result, e=expected),
        )

    def test_calculate_hash(self):
        """
        Scenario checks calculate_hash function.
        Should return hash as a string.
        """

        sha256_hash = hashlib.sha256()
        sha256_hash.update(b"test data")
        expected = sha256_hash.hexdigest()
        path = "path/to/file"
        with mock.patch(
            "builtins.open", mock.mock_open(read_data=b"test data")
        ) as mock_file:
            assert open("path/to/file").read() == b"test data"
            mock_file.assert_called_with("path/to/file")
            result = self.tfd_cursor._calculate_hash(path)

        self.assertEqual(
            result,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=result, e=expected),
        )

    @mock.patch("tensorflow.saved_model.load", return_values=object)
    def test_validate_model_or_module(self, tf_patch):
        """
        Scenario tests validate_model function.
        If everything is ok should return nothing.
        """

        path = "/path/to/file"
        result = None
        try:
            with mock.patch("os.path", return_values=True):
                with mock.patch("os.listdir", return_value=["README.md"]):
                    self.tfd_cursor._validate_model_or_module(path)
        except Exception as e:
            result = e
        self.assertIsNone(result)

    @mock.patch("tensorflow.saved_model.load", return_values=object)
    def test_validate_model_or_module_noreadme_err(self, tf_patch):
        """
        Scenario tests validate_model_or_module function.
        Should return error when there is no readme.md file.
        """
        path = "/path/to/file"
        result = None
        try:
            with mock.patch("os.path", return_values=True):
                with mock.patch("os.listdir", return_value=["test1.txt, test2.txt"]):
                    self.tfd_cursor._validate_model_or_module(path)
        except ValueError as e:
            result = e
        expected = "Directory without README.md file!"
        self.assertEqual(
            expected,
            str(result),
            msg="Got '{r}', expected '{e}'".format(r=str(result), e=expected),
        )
        self.assertIsNotNone(result)

    @mock.patch("tensorflow.saved_model.load", return_values=object)
    def test_validate_model_or_module_nofiles_err(self, tf_patch):
        """
        Scenario tests validate_model_or_module when there's no files in given directory.
        Should return Error.
        """

        path = "/path/to/file"
        result = None
        try:
            self.tfd_cursor._validate_model_or_module(path)
        except Exception as e:
            result = e
        self.assertIsNotNone(result)

    def test_validate_model_or_module_tfload_err(self):
        """
        Scenario tests if validate_model_or_module function returns valid response when got error on TF load.
        """
        path = "/path/to/file"
        result = None
        try:
            self.tfd_cursor._validate_model_or_module(path)
        except ValueError as e:
            result = e
        expected = "TensorFlow model validation failed! Error: SavedModel file does not exist at: /path/to/file/{saved_model.pbtxt|saved_model.pb}"
        self.assertEqual(
            expected,
            str(result),
            msg="Got '{r}', expected '{e}'".format(r=str(result), e=expected),
        )
        self.assertIsNotNone(result)

    @mock.patch("tarfile.open", return_values=object)
    @mock.patch("tensorflow.saved_model.load", return_values=object)
    def test_validate_archived_model_or_module(self, tar_mock, tf_mock):
        """
        Scenario tests validate_archived_model_or_module function.
        None on return when everything is ok.
        """

        err = None
        path_tar = "/path/to/file.tar"
        with mock.patch("os.path", return_values=True):
            with mock.patch("os.listdir", return_value=["README.md"]):
                try:
                    self.tfd_cursor._validate_archived_model_or_module(path_tar)
                except Exception as e:
                    err = e
                    pass

        self.assertIsNone(err, msg="Expected None, got '{e}'".format(e=err))

    @mock.patch("tarfile.open", return_values=object)
    @mock.patch("tensorflow.saved_model.load", return_values=object)
    def test_validate_archived_model_or_module_err(self, tar_mock, tf_mock):
        """
        Scenario tests validate_archived_model_or_module function when something is wrong with TFD Api.
        Should return error.
        """

        err = None
        path_tar = "/path/to/file.tar"
        with mock.patch("os.path", return_values=True):
            with mock.patch("os.listdir", return_value=["test1.txt"]):
                try:
                    self.tfd_cursor._validate_archived_model_or_module(path_tar)
                except Exception as e:
                    err = e
                    pass
        expected = "Directory without README.md file!"
        self.assertEqual(
            expected,
            str(err),
            msg="Got '{r}', expected '{e}'".format(r=str(err), e=expected),
        )

    @mock.patch("tarfile.open", return_values=object)
    def test_extract_archive(self, tar_mock):
        """
        Scenario tests extract_archive function. Should return None if ok.
        Warning: it does not consider the internal tar_filter function.
        """

        src = "/path/to/file.tar"
        dst = "/path/to/extraction"
        err = None
        try:
            self.tfd_cursor._extract_archive(src, dst)
        except Exception as e:
            err = e
            pass
        self.assertIsNone(err, msg="Expected None, got '{r}'".format(r=err))

    @mock.patch.object(tarfile, "open", autospec=True)
    def test_create_archive(self, tarfile_mock):
        """
        Scenario tests create_archive function. Should return string with hash of archived files.
        Warning: it does not consider the internal tar_filter function.
        """

        src = "/path/to/files/"
        dst = "/path/to/create/archive"
        sha256_hash = hashlib.sha256()
        sha256_hash.update(b"test data")
        expected = sha256_hash.hexdigest()
        with mock.patch(
            "builtins.open", mock.mock_open(read_data=b"test data")
        ) as mock_file:
            assert open(dst).read() == b"test data"
            mock_file.assert_called_with(dst)
            with mock.patch("os.path", return_values=True):
                _hash = self.tfd_cursor.create_archive(src, dst)

        self.assertEqual(expected, _hash)

    @requests_mock.mock()
    @mock.patch("tarfile.open", return_values=object)
    @mock.patch("pathlib.Path.open")
    @mock.patch("tensorflow.saved_model.load", return_values=object)
    @mock.patch("os.remove", return_values=None)
    @mock.patch("pathlib.Path.is_dir", return_value=False)
    def test_upload_model_tararchive(
        self, requests_mock, path_mock, tar_mock, tf_mock, rm_mock, isdir_mock
    ):
        """
        Scenario tests upload_model function.
        Given path is tar archive file. Should return string with upload success message.
        """

        path = "/dir/to/test/test_model.tar"
        url = endpoint["label"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )
        requests_mock.post(
            url,
            text="ok",
            status_code=200,
        )
        expected = "Upload success!"
        with mock.patch("os.listdir", return_value=["README.md", "test_model.tar"]):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=b"test data")
            ) as mock_file:
                assert open(path).read() == b"test data"
                mock_file.assert_called_with(path)
                result = self.tfd_cursor.upload_model(path)

        self.assertEqual(
            result, expected, msg="Got {r}, expected {e}".format(r=result, e=expected)
        )

    @requests_mock.mock()
    @mock.patch("tarfile.open", return_values=object)
    @mock.patch("pathlib.Path.open", return_values="file.content")
    @mock.patch("tensorflow.saved_model.load", return_values=object)
    @mock.patch("os.remove", return_values=None)
    @mock.patch("pathlib.Path.is_dir", return_value=False)
    def test_upload_model_tararchive_err(
        self,
        requests_mock,
        path_mock,
        tar_mock,
        tf_mock,
        rm_mock,
        isdir_mock,
        timeout: int = 120,
    ):
        """
        Scenario tests upload_model function with tar archive file as path param when response code from TFD Api is not 200.
        Should return error message.
        """

        path = "/dir/to/test/test_model.tar"
        url = endpoint["label"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )
        err_msg = "Model can't be uploaded"
        requests_mock.post(
            url,
            text=err_msg,
            status_code=500,
        )

        expected = "Upload failed!\nServer response: {r}".format(r=err_msg)
        with mock.patch("os.listdir", return_value=["README.md", "test_model.tar"]):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=b"test data")
            ) as mock_file:
                assert open(path).read() == b"test data"
                mock_file.assert_called_with(path)
                result = self.tfd_cursor.upload_model(path)

        self.assertEqual(
            result,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=result, e=expected),
        )

    @mock.patch("pathlib.Path.is_dir", return_value=False)
    def test_upload_model_extension_err(self, is_dir_mock):
        """
        Scenario tests upload_model function with wrong param (not tar archive extensions).
        Should return extension error.
        """

        path = "/dir/to/test/test_model.not_tar"
        expected = "Unexpected file extension. src_path must be a tar archive"
        result = self.tfd_cursor.upload_model(path)
        self.assertEqual(
            result,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=result, e=expected),
        )

    @requests_mock.mock()
    @mock.patch("tarfile.open", return_values=object)
    @mock.patch("pathlib.Path.open", return_values="file.content")
    @mock.patch("tensorflow.saved_model.load", return_values=object)
    @mock.patch("os.remove", return_values=None)
    @mock.patch("pathlib.Path.is_dir", return_value=True)
    @mock.patch("pathlib.Path.unlink", return_value=None)
    @mock.patch.object(tarfile, "open", autospec=True)
    def test_upload_model_dir(
        self,
        requests_mock,
        path_mock,
        tar_mock,
        tf_mock,
        rm_mock,
        is_dir_mock,
        unlink_mock,
        tarfile_mock,
        timeout: int = 120,
    ):
        """
        Scenario tests upload_model function with directory as param.
        Should return success message.
        """

        path = "/dir/to/test/"
        url = endpoint["label"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )
        requests_mock.post(
            url,
            text="ok",
            status_code=200,
        )
        expected = "Upload success!"
        with mock.patch("os.listdir", return_value=["README.md"]):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=b"test data")
            ) as mock_file:
                f_name = "tmp_upload_{t}.tar".format(t=int(time.time()))
                assert open(f_name).read() == b"test data"
                mock_file.assert_called_with(f_name)
                result = self.tfd_cursor.upload_model(path)

        self.assertEqual(
            result,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=result, e=expected),
        )

    @requests_mock.mock()
    @mock.patch("tarfile.open", return_values=object)
    @mock.patch("pathlib.Path.open", return_values="file.content")
    @mock.patch("tensorflow.saved_model.load", return_values=object)
    @mock.patch("os.remove", return_values=None)
    @mock.patch("pathlib.Path.is_dir", return_value=True)
    @mock.patch("pathlib.Path.unlink", return_value=None)
    @mock.patch.object(tarfile, "open", autospec=True)
    def test_upload_model_dir_err(
        self,
        requests_mock,
        path_mock,
        tar_mock,
        tf_mock,
        rm_mock,
        is_dir_mock,
        unlink_mock,
        tarfile_mock,
        timeout: int = 120,
    ):
        """
        Scenario tests upload_model function with directory as param when api responded with code other than 200.
        Should return error.
        """

        path = "/dir/to/test/"
        url = endpoint["label"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )
        err_msg = "Can't upload model"
        requests_mock.post(
            url,
            text=err_msg,
            status_code=500,
        )
        expected = "Upload failed!\nServer response: {r}".format(r=err_msg)
        with mock.patch("os.listdir", return_value=["README.md"]):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=b"test data")
            ) as mock_file:
                f_name = "tmp_upload_{t}.tar".format(t=int(time.time()))
                assert open(f_name).read() == b"test data"
                mock_file.assert_called_with(f_name)
                result = self.tfd_cursor.upload_model(path)

        self.assertEqual(
            result,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=result, e=expected),
        )

    @requests_mock.mock()
    @mock.patch("tarfile.open", return_values=object)
    @mock.patch("pathlib.Path.open", return_values="blabla")
    @mock.patch("tensorflow.saved_model.load", return_values=object)
    @mock.patch("os.remove", return_values=None)
    @mock.patch("pathlib.Path.is_dir", return_value=False)
    def test_upload_module_tararchive(
        self,
        requests_mock,
        path_mock,
        tar_mock,
        tf_mock,
        rm_mock,
        isdir_mock,
        timeout: int = 120,
    ):
        """
        Scenario tests upload_module function.
        Should return success message.
        """
        path = "/dir/to/test/test_module.tar"
        url = endpoint["modules"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
        )
        requests_mock.post(
            url,
            text="Module uploaded",
            status_code=200,
        )
        expected = "Upload success!"
        with mock.patch("os.listdir", return_value=["README.md", "test_module.tar"]):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=b"test data")
            ) as mock_file:
                assert open(path).read() == b"test data"
                mock_file.assert_called_with(path)
                result = self.tfd_cursor.upload_module(path)
        self.assertEqual(
            result,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=result, e=expected),
        )

    @requests_mock.mock()
    @mock.patch("tarfile.open", return_values=object)
    @mock.patch("pathlib.Path.open", return_values="file.content")
    @mock.patch("tensorflow.saved_model.load", return_values=object)
    @mock.patch("os.remove", return_values=None)
    @mock.patch("pathlib.Path.is_dir", return_value=False)
    def test_upload_module_tararchive_err(
        self,
        requests_mock,
        path_mock,
        tar_mock,
        tf_mock,
        rm_mock,
        isdir_mock,
        timeout: int = 120,
    ):
        """
        Scenario tests upload_module function when tar aerchive raises error.
        Should return error.
        """

        path = "/dir/to/test/test_module.tar"
        url = endpoint["modules"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
        )
        err_msg = "Module can't be uploaded"
        requests_mock.post(
            url,
            text=err_msg,
            status_code=500,
        )

        expected = "Upload failed!\nServer response: {r}".format(r=err_msg)
        with mock.patch("os.listdir", return_value=["README.md", "test_module.tar"]):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=b"test data")
            ) as mock_file:
                assert open(path).read() == b"test data"
                mock_file.assert_called_with(path)
                result = self.tfd_cursor.upload_module(path)
        self.assertEqual(
            result, expected, msg="Got {r}, expected {e}".format(r=result, e=expected)
        )

    @mock.patch("pathlib.Path.is_dir", return_value=False)
    def test_upload_module_extension_err(self, is_dir_mock):
        """
        Scenario tests upload_module function when got bad file extension.
        Should return error.
        """

        path = "/dir/to/test/test_module.not_tar"
        expected = "Unexpected file extension. src_path must be a tar archive"
        result = self.tfd_cursor.upload_module(path)
        self.assertEqual(
            result,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=result, e=expected),
        )

    @requests_mock.mock()
    @mock.patch("tarfile.open", return_values=object)
    @mock.patch("pathlib.Path.open", return_values="file.content")
    @mock.patch("tensorflow.saved_model.load", return_values=object)
    @mock.patch("os.remove", return_values=None)
    @mock.patch("pathlib.Path.is_dir", return_value=True)
    @mock.patch("pathlib.Path.unlink", return_value=None)
    @mock.patch.object(tarfile, "open", autospec=True)
    def test_upload_module_dir(
        self,
        requests_mock,
        path_mock,
        tar_mock,
        tf_mock,
        rm_mock,
        is_dir_mock,
        unlink_mock,
        tarfile_mock,
        timeout: int = 120,
    ):
        """
        Scenario tests upload_module function with directory as param.
        Should return success message.
        """

        path = "/dir/to/test/"
        url = endpoint["modules"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
        )
        requests_mock.post(
            url,
            text="Module uploaded",
            status_code=200,
        )
        expected = "Upload success!"
        with mock.patch("os.listdir", return_value=["README.md"]):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=b"test data")
            ) as mock_file:
                f_name = "tmp_upload_{t}.tar".format(t=int(time.time()))
                assert open(f_name).read() == b"test data"
                mock_file.assert_called_with(f_name)
                result = self.tfd_cursor.upload_module(path)
        self.assertEqual(
            result,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=result, e=expected),
        )

    @requests_mock.mock()
    @mock.patch("tarfile.open", return_values=object)
    @mock.patch("pathlib.Path.open", return_values="file.content")
    @mock.patch("tensorflow.saved_model.load", return_values=object)
    @mock.patch("os.remove", return_values=None)
    @mock.patch("pathlib.Path.is_dir", return_value=True)
    @mock.patch("pathlib.Path.unlink", return_value=None)
    @mock.patch.object(tarfile, "open", autospec=True)
    def test_upload_module_dir_err(
        self,
        requests_mock,
        path_mock,
        tar_mock,
        tf_mock,
        rm_mock,
        is_dir_mock,
        unlink_mock,
        tarfile_mock,
        timeout: int = 120,
    ):
        """
        Scenario tests upload_module function when dir is invalid.
        Should return directory error.
        """

        path = "/dir/to/test/"
        url = endpoint["modules"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
        )
        err_msg = "Can't upload module"
        requests_mock.post(
            url,
            text=err_msg,
            status_code=500,
        )
        expected = "Upload failed!\nServer response: {r}".format(r=err_msg)
        with mock.patch("os.listdir", return_value=["README.md"]):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=b"test data")
            ) as mock_file:
                f_name = "tmp_upload_{t}.tar".format(t=int(time.time()))
                assert open(f_name).read() == b"test data"
                mock_file.assert_called_with(f_name)
                result = self.tfd_cursor.upload_module(path)
        self.assertEqual(
            result,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=result, e=expected),
        )

    @requests_mock.mock()
    @mock.patch("tarfile.open", return_values=object)
    @mock.patch("pathlib.Path.open", return_values="file.content")
    @mock.patch("tensorflow.saved_model.load", return_values=object)
    @mock.patch("os.remove", return_values=None)
    @mock.patch("pathlib.Path.is_dir", return_value=False)
    def test_deploy_model_label_tararchive(
        self,
        requests_mock,
        tar_mock,
        pathopen_mock,
        tf_load_mock,
        osrm_mock,
        is_dir_mock,
    ):
        """
        Scenario tests deploy_model function with params: label and tar archive file.
        Should return success message.
        """

        path = "/dir/to/test/test_model.tar"
        url = endpoint["label"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )
        requests_mock.post(
            url,
            text="ok",
            status_code=200,
        )
        url_reload = endpoint["reload"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            reload_type=True,
        )
        requests_mock.post(
            url_reload,
            text="",
            status_code=200,
        )
        with mock.patch("os.listdir", return_value=["README.md", "test_model.tar"]):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=b"test data")
            ) as mock_file:
                assert open(path).read() == b"test data"
                mock_file.assert_called_with(path)
                response = self.tfd_cursor.deploy_model(path)
        expected = "Deploy results:\nupload: {}\nreload: {}".format(
            "Upload success!", "reload_config success!"
        )
        self.assertEqual(
            response,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=response, e=expected),
        )

    @requests_mock.mock()
    @mock.patch("tarfile.open", return_values=object)
    @mock.patch("pathlib.Path.open", return_values="blabla")
    @mock.patch("tensorflow.saved_model.load", return_values=object)
    @mock.patch("os.remove", return_values=None)
    @mock.patch("pathlib.Path.is_dir", return_value=False)
    def test_deploy_model_label_tararchive_err(
        self,
        requests_mock,
        tar_mock,
        pathopen_mock,
        tf_load_mock,
        osrm_mock,
        is_dir_mock,
    ):
        """
        Scenario tests deploy_model function with given label as a param when tar archive receives an error.
        Should return error.
        """

        path = "/dir/to/test/test_model.tar"
        url = endpoint["label"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            name=self.name,
            label=self.label,
        )
        err_msg = "error messgae"
        requests_mock.post(
            url,
            text=err_msg,
            status_code=500,
        )
        url_reload = endpoint["reload"].format(
            host=self.host,
            port=self.port,
            team=self.team,
            project=self.project,
            reload_type=True,
        )
        requests_mock.post(
            url_reload,
            text="",
            status_code=200,
        )
        with mock.patch("os.listdir", return_value=["README.md", "test_model.tar"]):
            with mock.patch(
                "builtins.open", mock.mock_open(read_data=b"test data")
            ) as mock_file:
                assert open(path).read() == b"test data"
                mock_file.assert_called_with(path)
                response = self.tfd_cursor.deploy_model(path)
        expected = "Deploy failed. Upload error: {e}".format(
            e="Upload failed!\nServer response: " + err_msg
        )
        self.assertEqual(
            response,
            expected,
            msg="Got '{r}', expected '{e}'".format(r=response, e=expected),
        )

    def test_generate_model_readme_nometrics(self):
        """
        Scenario tests generate_model_readme function if metrics were not given.
        Function returns nothing and there should be no error.
        """

        description = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        dst_path = "/path/to/files/"
        err = None

        with mock.patch("builtins.open", mock.mock_open(read_data="")) as mock_file:
            assert open("/path/to/files/README.md").read() == ""
            mock_file.assert_called_with("/path/to/files/README.md")
            try:
                self.tfd_cursor.generate_model_readme(dst_path, description)
            except Exception as e:
                err = e
        self.assertIsNone(err, msg="Expected None, got '{e}'".format(e=err))

    def test_generate_model_readme_metrics(self):
        """
        Scenario tests generate_model function with metrics as param.
        Function returns nothing and there should be no error.
        """

        description = """Lorem ipsum dolor sit amet,
        consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
        Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
        Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
        """
        metrics = {"test1": 1, "test2": 2, "test3": 3}
        dst_path = "/path/to/files/"
        err = None

        with mock.patch("builtins.open", mock.mock_open(read_data="")) as mock_file:
            assert open("/path/to/files/README.md").read() == ""
            mock_file.assert_called_with("/path/to/files/README.md")
            try:
                self.tfd_cursor.generate_model_readme(dst_path, description, metrics)
            except Exception as e:
                err = e
        self.assertIsNone(err, msg="Expected None, got '{e}'".format(e=err))

    def test_generate_module_readme(self):
        """
        Scenario tests generate_module_readme function.
        Function returns nothing and there should be no error.
        """

        description = """Lorem ipsum dolor sit amet,
        consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
        Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
        Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
        """
        dst_path = "/path/to/files/"
        err = None
        with mock.patch("builtins.open", mock.mock_open(read_data="")) as mock_file:
            assert open("/path/to/files/README.md").read() == ""
            mock_file.assert_called_with("/path/to/files/README.md")
            try:
                self.tfd_cursor.generate_module_readme(dst_path, description)
            except Exception as e:
                err = e
        self.assertIsNone(err, msg="Expected None, got '{e}'".format(e=err))


if __name__ == "__main__":
    unittest.main()
