# TensorFlow Deploy Utils
Utilities for communication and easy management of [TensorFlow Deploy](https://github.com/grupawp/tensorflow-deploy)

![build](https://github.com/grupawp/tensorflow-deploy-utils/workflows/Build%20and%20test/badge.svg)

*TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.*

## Installation
```bash
pip install tensorflow-deploy-utils
```

## Usage
```python
import tensorflow_deploy_utils as tfd

tfd_cursor = tfd.TFD(YOUR_TEAM, YOUR_PROJECT, YOUR_MODEL_NAME)
# now you can make some operations
tfd_cursor.deploy_model("path/to/your/model")
tfd_cursor.list_models()
tfd_cursor.get_config()
tfd_cursor.get_module("path/to/write/module")
tfd_cursor.set_label(3, "my_label")
tfd_cursor.set_stable(2)
# and more
```

## Building
```bash
python setup.py sdist bdist_wheel
pip install dist/tensorflow_deploy_utils-X.X.X-py3-none-any.whl
```