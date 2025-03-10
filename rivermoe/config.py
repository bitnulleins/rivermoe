import logging.config
import os

import dotenv

dotenv.load_dotenv()

dirname, filename = os.path.split(os.path.abspath(__file__))


class Config:
    """Configuration class for the riverMoE package as environment variables.

    Returns
    -------
    str
        String of all config parameters.
    """

    random_seed: int = os.getenv("RANDOM_SEED", 42)
    log_level: str = os.getenv("LOG_LEVEL", "WARNING")
    font_path: str = os.getenv("GDFONTPATH", False)
    font: str = os.getenv("GDFONT", None)

    def __str__(self):
        return f"Config(random_seed={self.random_seed}, log_level={self.log_level}, draw_font={self.font_path}), font={self.font})"


# Instantiate the config object
config = Config()

# Load logging configuration from logger.ini
logging.config.fileConfig(dirname + "/../logger.ini", disable_existing_loggers=False)

# Update the root logger's log level
logging.getLogger().setLevel(config.log_level.upper())
