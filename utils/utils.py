import tomllib


def read_config(toml_file: str) -> dict:
    with open(f"config/{toml_file}.toml", "rb") as f:
        data = tomllib.load(f)
    return data
