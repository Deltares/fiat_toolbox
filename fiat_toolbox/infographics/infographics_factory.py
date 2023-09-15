from pathlib import Path
from typing import Union

from fiat_toolbox.infographics.infographics import InfographicsParser
from fiat_toolbox.infographics.infographics_interface import IInfographicsParser
from fiat_toolbox.infographics.risk_infographics import RiskInfographicsParser


class InforgraphicFactory:
    @staticmethod
    def create_infographic_file_writer(
        infographic_mode: str,
        scenario_name: str,
        metrics_full_path: Union[Path, str],
        config_base_path: Union[Path, str],
        output_base_path: Union[Path, str],
    ) -> IInfographicsParser:
        """
        Create a infographic file writer.

        Parameters
        ----------
        infographic_mode : str
            The mode of the infographic file writer to create.
        config_file : Path
            The path to the infographic file.

        Returns
        -------
        IInfographicsFileWriter
            A infographic file writer.
        """
        if infographic_mode == "single_event":
            return InfographicsParser(
                scenario_name, metrics_full_path, config_base_path, output_base_path
            )
        elif infographic_mode == "risk":
            return RiskInfographicsParser(
                scenario_name, metrics_full_path, config_base_path, output_base_path
            )
        else:
            raise ValueError(f"Infographic_mode {infographic_mode} not supported")
