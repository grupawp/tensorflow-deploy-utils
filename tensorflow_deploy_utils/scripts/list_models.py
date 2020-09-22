import argparse
from tensorflow_deploy_utils.TFD import TFD


def main() -> None:

    parser = argparse.ArgumentParser(description="Script list models for given criteria")
    parser.add_argument("--host", type=str, default="localhost.service", help="TensorFlow Deploy instance IP or address")
    parser.add_argument("--port", type=int, default=9500, help="TensorFlow Deploy instance port")
    parser.add_argument("--team", type=str, required=False, default="", help="TEAM")
    parser.add_argument("--project", type=str, required=False, default="", help="PROJECT")
    parser.add_argument("--name", type=str, required=False, default="", help="NAME")
    parser.add_argument("--version", type=int, required=False, default=0, help="Model version")
    parser.add_argument("--label", type=str, required=False, default="", help="Model label")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")

    args = parser.parse_args()

    tfd_cursor = TFD(team=args.team, project=args.project, name=args.name, host=args.host, port=args.port,
                     verbose=args.verbose)
    result_df = tfd_cursor.list_models(team=args.team, project=args.project, name=args.name, version=args.version,
                                       label=args.label)
    print(result_df)


if __name__ == "__main__":

    main()
