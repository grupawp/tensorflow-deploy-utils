import argparse
from tensorflow_deploy_utils.TFD import TFD


def main() -> None:

    parser = argparse.ArgumentParser(description="Script create tar archive with TF model or module compatible with "
                                                 "tensorflow-deploy")
    parser.add_argument("src_path", type=str, help="Source path to file or dir to archive - this must be")
    parser.add_argument("dst_path", type=str, help="Destination path to write archive")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")

    args = parser.parse_args()

    tfd_cursor = TFD(**args.__dict__)
    print(tfd_cursor.create_archive(args.src_path, args.dst_path))


if __name__ == "__main__":

    main()
