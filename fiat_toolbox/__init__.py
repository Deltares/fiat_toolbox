__version__ = "0.1.14"

from packaging import version

class FiatColumns:
    """Object with mapping of FIAT attributes to columns names."""

    def __init__(self, fiat_version:str="0.2"):
        self.version = version.parse(fiat_version)
        self.set_attributes()

    def set_attributes(self):
        if self.version > version.parse("0.1"):
            self.object_id = "object_id"
            self.object_name = "object_name"
            self.primary_object_type = "primary_object_type"
            self.secondary_object_type = "secondary_object_type"
            self.extraction_method = "extract_method"
            self.ground_floor_height = "ground_flht"
            self.ground_elevation = "ground_elevtn"
            self.fn_damage = "fn_damage_"
            self.max_potential_damage = "max_damage_"
            self.aggregation_label = "aggregation_label:"
            self.inundation_depth = "inun_depth"
            self.damage = "damage_"
            self.total_damage = "total_damage"
            self.ead_damage = "ead_damage"
        elif self.version == version.parse("0.1.0rc2"):
            self.object_id = "Object ID"
            self.object_name = "Object Name"
            self.primary_object_type = "Primary Object Type"
            self.secondary_object_type = "Secondary Object Type"
            self.extraction_method = "Extraction Method"
            self.ground_floor_height = "Ground Flood Height"
            self.ground_elevation = "Ground Elevation"
            self.fn_damage = "Damage Function: "
            self.max_potential_damage = "Max Potential Damage: "
            self.aggregation_label = "Aggregation Label: "
            self.inundation_depth = "Inundation Depth"
            self.damage = "Damage: "
            self.total_damage = "Total Damage"
            self.ead_damage = "Risk (EAD)"
        else:
            raise ValueError(f"Unsupported version: {self.version}")