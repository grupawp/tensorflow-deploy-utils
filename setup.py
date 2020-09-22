import setuptools


VERSION = "1.0.0"


with open("README.md", "r") as fh:
    readme = fh.read().strip()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().strip().split("\n")

with open("tensorflow_deploy_utils/version.py", "w") as fh:
    fh.write('VERSION="{v}"\n'.format(v=VERSION))


setuptools.setup(
    name="tensorflow_deploy_utils",
    version=VERSION,
    author="Wirtualna Polska Media S.A.",
    author_email="tfd@grupawp.pl",
    description="Utils for managing and communication with TensorFlow Deploy",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/grupawp/tensorflow-deploy-utils",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    license="ISC",
    keywords="tensorflow ai ml machine learning production serving kubernetes deploy tf tfd tfs",
    entry_points={
        "console_scripts": [
            "tfd_create_archive=tensorflow_deploy_utils.scripts.create_archive:main",
            "tfd_delete_label=tensorflow_deploy_utils.scripts.delete_label:main",
            "tfd_delete_model=tensorflow_deploy_utils.scripts.delete_model:main",
            "tfd_delete_module=tensorflow_deploy_utils.scripts.delete_module:main",
            "tfd_deploy_model=tensorflow_deploy_utils.scripts.deploy_model:main",
            "tfd_get_config=tensorflow_deploy_utils.scripts.get_config:main",
            "tfd_get_model=tensorflow_deploy_utils.scripts.get_model:main",
            "tfd_get_module=tensorflow_deploy_utils.scripts.get_module:main",
            "tfd_list_models=tensorflow_deploy_utils.scripts.list_models:main",
            "tfd_list_modules=tensorflow_deploy_utils.scripts.list_modules:main",
            "tfd_reload_config=tensorflow_deploy_utils.scripts.reload_config:main",
            "tfd_set_label=tensorflow_deploy_utils.scripts.set_label:main",
            "tfd_set_stable=tensorflow_deploy_utils.scripts.set_stable:main",
            "tfd_upload_model=tensorflow_deploy_utils.scripts.upload_model:main",
            "tfd_upload_module=tensorflow_deploy_utils.scripts.upload_module:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: ISC License (ISCL)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    project_urls={
        "Documentation": "https://github.com/grupawp/tensorflow-deploy-utils",
        "Source": "https://github.com/grupawp/tensorflow-deploy-utils",
        "Bug Report": "https://github.com/grupawp/tensorflow-deploy-utils/issues",
    },
)
