import argparse
from tensorflow_deploy_utils.TFD import TFD


def main() -> None:

    parser = argparse.ArgumentParser(description="Script upload compressed od uncompressed TF modules to TensorFlow Deploy")

    parser.add_argument("--host", type=str, default="localhost.service", help="TensorFlow Deploy instance IP or address")
    parser.add_argument("--port", type=int, default=9500, help="TensorFlow Deploy instance port")

    parser.add_argument("--path", type=str, required=True, help="Full path to module/module dir/archive")
    parser.add_argument("--team", type=str, required=True, help="TEAM")
    parser.add_argument("--project", type=str, required=True, help="PROJECT")
    parser.add_argument("--name", type=str, required=True, help="NAME")
    parser.add_argument("--timeout", type=int, required=False, default=600, help="Upload timeout in seconds")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    tfd_cursor = TFD(**args.__dict__)
    print(tfd_cursor.upload_module(args.path, args.timeout))


if __name__ == "__main__":

    main()
