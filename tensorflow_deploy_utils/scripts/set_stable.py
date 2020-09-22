import argparse
from tensorflow_deploy_utils.TFD import TFD


def main() -> None:

    parser = argparse.ArgumentParser(description="Script set 'stable' label for given model")
    parser.add_argument("--host", type=str, default="localhost.service", help="TensorFlow Deploy instance IP or address")
    parser.add_argument("--port", type=int, default=9500, help="TensorFlow Deploy instance port")
    parser.add_argument("--team", type=str, required=True, help="TEAM")
    parser.add_argument("--project", type=str, required=True, help="PROJECT")
    parser.add_argument("--name", type=str, required=True, help="NAME")
    parser.add_argument("--version", type=int, required=True, help="Model VERSION")
    parser.add_argument("--retries", type=int, required=False, default=3, help="Number of retries in error case")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")

    args = parser.parse_args()

    tfd_cursor = TFD(**args.__dict__)
    print(tfd_cursor.set_stable(args.version, args.retries))


if __name__ == "__main__":

    main()
