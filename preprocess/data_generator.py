import argparse
import importlib

from soundsleep.utils.training import set_seed

parser = argparse.ArgumentParser(description="Data Generator")

# Config for processing option
parser.add_argument(
    "-a2n", "--audio_to_numpy", action="store_true", help="Process audio data and save into numpy array"
)

# Config for dataset
parser.add_argument("-d", "--data_name", type=str, required=True, metavar="DATA", help="Data name to generate")
parser.add_argument("-p", "--portion", type=float, default=0.7, metavar="N", help="Training set portion")


class BaseDataGenerator:
    """Data Generator classes in soundsleep.preprocess must inherit this base class."""

    def __init__(self):
        self.args = None

    def generate_data(self):
        """Main routine for data generation."""
        if self.args.audio_to_numpy:
            self.audio2numpy()

    def video2audio(self):
        raise NotImplementedError

    def audio2numpy(self):
        raise NotImplementedError


def get_data_generator(data_name, parser):
    """Get preprocessing class instance by dynamically importing."""
    module_path = ".".join(["soundsleep.preprocess", data_name, data_name])
    module = importlib.import_module(module_path)
    DataGen = getattr(module, data_name)
    return DataGen(parser)


if __name__ == "__main__":
    # Set random seeds
    set_seed(0)

    # Parse known args
    args = parser.parse_known_args()[0]

    # Get preprocessing class instance
    data_generator = get_data_generator(args.data_name, parser)

    # Generate data
    data_generator.generate_data()
