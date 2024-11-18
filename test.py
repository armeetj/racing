import os


def main():
    if "VISION" in os.environ and os.environ["VISION"] == "1":
        print("VISION is set to 1.")
    else:
        print("VISION is not set to 1.")


if __name__ == "__main__":
    main()
