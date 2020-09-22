import argparse
from tensorflow_deploy_utils.TFD import TFD


def main() -> None:

    parser = argparse.ArgumentParser(description="Script download and write specific model in destination path")
    parser.add_argument("--dst_path", type=str, required=True, help="Path where write model")
    parser.add_argument("--host", type=str, default="localhost.service", help="TensorFlow Deploy instance IP or address")
    parser.add_argument("--port", type=int, default=9500, help="TensorFlow Deploy instance port")
    parser.add_argument("--team", type=str, required=True, help="TEAM")
    parser.add_argument("--project", type=str, required=True, help="PROJECT")
    parser.add_argument("--name", type=str, required=True, help="NAME")
    parser.add_argument("--version", type=int, required=False, default=0, help="Version for given model")
    parser.add_argument("--label", type=str, required=False, default="", help="Label for given model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")

    args = parser.parse_args()

    tfd_cursor = TFD(**args.__dict__)
    if args.version:
        print(tfd_cursor.get_model(dst_path=args.dst_path, version=args.version))
    if args.label:
        print(tfd_cursor.get_model(dst_path=args.dst_path, label=args.label))


if __name__ == "__main__":

    main()
