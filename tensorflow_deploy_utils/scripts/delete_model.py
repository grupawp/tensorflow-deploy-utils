import argparse
from tensorflow_deploy_utils.TFD import TFD


def main() -> None:

    parser = argparse.ArgumentParser(description="Script remove specific model from TensorFlow Deploy")
    parser.add_argument("--host", type=str, default="localhost.service", help="TensorFlow Deploy instance IP or address")
    parser.add_argument("--port", type=int, default=9500, help="TensorFlow Deploy instance port")
    parser.add_argument("--team", type=str, required=True, help="TEAM")
    parser.add_argument("--project", type=str, required=True, help="PROJECT")
    parser.add_argument("--name", type=str, required=True, help="NAME")
    parser.add_argument("--version", type=int, default=None, help="Model VERSION")
    parser.add_argument("--label", type=int, default="", help="Model LABEL")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")

    args = parser.parse_args()
    label = args.label
    version = args.version

    tfd_cursor = TFD(**args.__dict__)
    if (version is not None and label) or (version is None and not label):
        print("One of two parameters must be given: version or label")

    if args.label:
        print(tfd_cursor.delete_model(label))
    elif args.version is not None:
        print(tfd_cursor.delete_model(version))
    


if __name__ == "__main__":

    main()
