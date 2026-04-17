__version__ = "0.1.24"

from packaging import version
from pydantic import BaseModel


class FiatColumns(BaseModel):
    """
    Model defining the FIAT column types and their naming format.
    All attributes are strings that can be:
    - static: with a standard name, e.g. 'object_id'
    - dynamic: with wildcard parts, e.g. 'max_damage_{name}' or 'damage_{name}_{years}y'
    """

    object_id: str
    object_name: str
    primary_object_type: str
    secondary_object_type: str
    extraction_method: str
    ground_floor_height: str
    ground_elevation: str
    damage_function: str
    max_potential_damage: str
    aggregation_label: str
    inundation_depth: str
    inundation_depth_rp: str
    reduction_factor: str
    reduction_factor_rp: str
    damage: str
    damage_rp: str
    total_damage: str
    total_damage_rp: str
    risk_ead: str
    segment_length: str  # TODO should this be here since it is not a FIAT attribute?


def get_fiat_columns(fiat_version: str = "0.2") -> FiatColumns:
    """
    Returns the column mappings for different versions of  FIAT.
    Parameters:
    fiat_version (str): The version of the FIAT. Default is "0.2".
    Returns:
    FiatColumns: An instance of FiatColumns with the appropriate column mappings for the specified version.
    Raises:
    ValueError: If the specified version is not supported.
    Supported Versions:
    - "1.0" and greater: Delft-FIAT v1 schema (single object_type; ground_flht +
      ground_elevtn remain as FloodAdapt working columns and a separate
      ``elevation`` column is materialised at write-time for the binary).
    - "0.2"–"0.2.x": v0.2 snake-case schema (primary/secondary_object_type,
      ground_flht + ground_elevtn).
    - "0.1.0rc2": Display-name schema.
    """
    fiat_version = version.parse(fiat_version)
    # Columns for versions >= 1.0 — Delft-FIAT v1 schema
    if fiat_version >= version.parse("1.0"):
        fiat_columns = FiatColumns(
            object_id="object_id",
            object_name="object_name",
            # v1 collapsed primary/secondary into a single object_type.
            primary_object_type="object_type",
            secondary_object_type="object_type",
            extraction_method="extract_method",
            # v1 FIAT reads floor height as "elevation" and ground elevation as
            # "reference" (the latter only for flood.level mode). Mapping these
            # directly via FiatColumns eliminates any materialization step.
            ground_floor_height="elevation",
            ground_elevation="reference",
            damage_function="fn_damage_{name}",
            max_potential_damage="max_damage_{name}",
            aggregation_label="aggregation_label:{name}",
            # v1 output columns use a hazard-band suffix: depth_<H>, damage_<suffix>_<H>,
            # total_damage_<H>. For flood.level the prefix is level_ instead of depth_.
            inundation_depth="depth_{hazard}",
            inundation_depth_rp="depth_{hazard}",
            reduction_factor="red_fact",
            reduction_factor_rp="red_fact_{hazard}",
            damage="damage_{name}_{hazard}",
            damage_rp="damage_{name}_{hazard}",
            total_damage="total_damage_{hazard}",
            total_damage_rp="total_damage_{hazard}",
            risk_ead="ead_{name}",
            segment_length="segment_length",
        )
    # Columns for 0.1 < version < 1.0 (v0.2 snake-case schema)
    elif fiat_version > version.parse("0.1"):
        fiat_columns = FiatColumns(
            object_id="object_id",
            object_name="object_name",
            primary_object_type="primary_object_type",
            secondary_object_type="secondary_object_type",
            extraction_method="extract_method",
            ground_floor_height="ground_flht",
            ground_elevation="ground_elevtn",
            damage_function="fn_damage_{name}",
            max_potential_damage="max_damage_{name}",
            aggregation_label="aggregation_label:{name}",
            inundation_depth="inun_depth",
            inundation_depth_rp="inun_depth_{years}y",
            reduction_factor="red_fact",
            reduction_factor_rp="red_fact_{years}y",
            damage="damage_{name}",
            damage_rp="damage_{name}_{years}y",
            total_damage="total_damage",
            total_damage_rp="total_damage_{years}y",
            risk_ead="ead_damage",
            segment_length="segment_length",
        )
    # Columns for version 0.1.0rc2
    elif fiat_version == version.parse("0.1.0rc2"):
        fiat_columns = FiatColumns(
            object_id="Object ID",
            object_name="Object Name",
            primary_object_type="Primary Object Type",
            secondary_object_type="Secondary Object Type",
            extraction_method="Extraction Method",
            ground_floor_height="Ground Floor Height",
            ground_elevation="Ground Elevation",
            damage_function="Damage Function: {name}",
            max_potential_damage="Max Potential Damage: {name}",
            aggregation_label="Aggregation Label: {name}",
            inundation_depth="Inundation Depth",
            inundation_depth_rp="Inundation Depth ({years}Y)",
            reduction_factor="Reduction Factor",
            reduction_factor_rp="Reduction Factor ({years}Y)",
            damage="Damage: {name}",
            damage_rp="Damage: {name} ({years}Y)",
            total_damage="Total Damage",
            total_damage_rp="Total Damage ({years}Y)",
            risk_ead="Risk (EAD)",
            segment_length="Segment Length",
        )
    else:
        raise ValueError(f"Unsupported version: {fiat_version}")

    return fiat_columns
