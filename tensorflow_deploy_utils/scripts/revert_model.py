import argparse
from tensorflow_deploy_utils.TFD import TFD


def main():

    parser = argparse.ArgumentParser(description="Script revert previous model stable version. It can be used only "
                                                 "ones, because remember just last stable version of the model.")
    parser.add_argument("--host", type=str, default="localhost.service", help="TensorFlow Deploy instance IP or address")
    parser.add_argument("--port", type=int, default=9500, help="TensorFlow Deploy instance port")
    parser.add_argument("--team", type=str, required=True, help="TEAM")
    parser.add_argument("--project", type=str, required=True, help="PROJECT")
    parser.add_argument("--name", type=str, required=True, help="NAME")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")

    args = parser.parse_args()

    tfd_cursor = TFD(**args.__dict__)
    print(tfd_cursor.revert_model())


if __name__ == "__main__":

    main()
