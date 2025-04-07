import os
import re
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import toml

from fiat_toolbox import get_fiat_columns


def _compile_pattern(pattern):
    """
    Compile a pattern with placeholders into a regex pattern.
    Args:
        pattern (str): The pattern containing placeholders in the format '{var}'.
    Returns:
        tuple: A tuple containing the compiled regex pattern and a list of placeholders.
    """
    # Escape special characters in pattern except for '{var}'
    escaped_pattern = re.escape(pattern)
    # Find all placeholders in the pattern
    placeholders = re.findall(r"\\{(.*?)\\}", escaped_pattern)
    # Replace placeholders with regex groups
    for placeholder in placeholders:
        escaped_pattern = escaped_pattern.replace(
            f"\\{{{placeholder}\\}}", f"(?P<{placeholder}>.*?)"
        )
    # Compile the regex pattern
    regex = re.compile(f"^{escaped_pattern}$")
    return regex, placeholders


def matches_pattern(string: str, pattern: str) -> bool:
    """
    Check if a string matches a pattern with placeholders.
    Args:
        string (str): The input string to be checked.
        pattern (str): The pattern containing placeholders in the format '{var}'.
    Returns:
        bool: True if the string matches the pattern, False otherwise.
    """
    regex, _ = _compile_pattern(pattern)
    return bool(regex.match(string))


def extract_variables(string: str, pattern: str) -> dict:
    """
    Extract variables from a string based on a pattern with placeholders.

    Args:
        string (str): The input string to be processed.
        pattern (str): The pattern containing placeholders in the format '{var}'.

    Returns:
        dict: A dictionary with the extracted variables and their values.
              If the pattern does not match the input string, an empty dictionary is returned.
    """
    regex, placeholders = _compile_pattern(pattern)

    # Find the match
    match = regex.match(string)
    if match:
        # Extract the captured groups into a dictionary
        extracted_vars = {
            placeholder: match.group(placeholder) for placeholder in placeholders
        }
        return extracted_vars
    return {}


def replace_pattern(string: str, pattern: str, replacement: str) -> str:
    """
    Replace placeholders in a string based on a pattern with a replacement string.
    Args:
        string (str): The input string to be processed.
        pattern (str): The pattern containing placeholders in the format '{var}'.
        replacement (str): The replacement string where placeholders will be replaced with corresponding values from the input string.
    Returns:
        str: The processed string with placeholders replaced by corresponding values from the input string.
             If the pattern does not match the input string, the original string is returned.
    """
    regex, placeholders = _compile_pattern(pattern)

    # Find the match
    match = regex.match(string)
    if match:
        # Replace placeholders in the replacement with the captured groups
        for placeholder in placeholders:
            replacement = replacement.replace(
                f"{{{placeholder}}}", match.group(placeholder)
            )
        return replacement
    return string


def convert_fiat(
    path_in: os.PathLike,
    path_out: os.PathLike,
    version_in: str = "0.1.0rc2",
    version_out: str = "0.2.1",
):
    """
    Converts FIAT data from one version to another by copying the input directory to the output directory,
    updating the settings file, and renaming columns in the exposure CSV file according to the specified versions.
    Args:
        path_in (os.PathLike): The input directory containing the FIAT data to be converted.
        path_out (os.PathLike): The output directory where the converted FIAT data will be saved.
        version_in (str, optional): The version of the input FIAT data. Defaults to "0.1.0rc2".
        version_out (str, optional): The version of the output FIAT data. Defaults to "0.2.1".
    Raises:
        FileNotFoundError: If the settings file or exposure CSV file is not found in the input directory.
        KeyError: If the expected keys are not found in the settings file.
    """
    path_in, path_out = Path(path_in), Path(path_out)
    if path_out.exists():
        shutil.rmtree(path_out)
    shutil.copytree(path_in, path_out)

    settings_path = path_out.joinpath("settings.toml")

    with open(settings_path, "r") as file:
        settings = toml.load(file)

    exposure_csv_path = settings_path.parent.joinpath(
        settings["exposure"]["csv"]["file"]
    )
    exposure_csv = pd.read_csv(exposure_csv_path)

    format_in = get_fiat_columns(fiat_version=version_in)
    format_out = get_fiat_columns(fiat_version=version_out)

    name_translation = {}
    for col in exposure_csv.columns:  # iterate through output columns
        for field in list(format_out.model_fields):  # check for each field
            fiat_col = getattr(format_in, field)
            if matches_pattern(col, fiat_col):
                impact_col = getattr(format_out, field)
                new_col = replace_pattern(col, fiat_col, impact_col)
                name_translation[col] = new_col  # save mapping

    # Rename exposure csv
    exposure_csv = exposure_csv.rename(columns=name_translation)
    exposure_csv.to_csv(exposure_csv_path, index=False)

    # Get geoms
    keys = [key for key in settings["exposure"]["geom"] if "file" in key]
    geoms_paths = [
        settings_path.parent.joinpath(settings["exposure"]["geom"][key]) for key in keys
    ]

    # Rename geoms
    for geom_path in geoms_paths:
        geom = gpd.read_file(geom_path)
        geom = geom.rename(columns=name_translation)
        geom_path.unlink()
        geom.to_file(geom_path)


