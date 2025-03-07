import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class IInfographicsParser(ABC):
    """Interface for creating the infographic"""

    logger: logging.Logger

    @abstractmethod
    def __init__(
        self,
        scenario_name: str,
        metrics_full_path: Union[Path, str],
        config_base_path: Union[Path, str],
        output_base_path: Union[Path, str],
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None: ...

    @abstractmethod
    def get_infographics(self) -> str:
        """Get the infographic for a scenario

        Returns
        -------
        str
            The infographic for the scenario as a string in html format
        """
        pass

    @abstractmethod
    def write_infographics_to_file() -> str:
        """Write the infographic for a scenario to file

        Returns
        -------
        str
            The path to the infographic file
        """
        pass

    @abstractmethod
    def get_infographics_html() -> str:
        """Get the path to the infographic html file

        Returns
        -------
        str
            The path to the infographic html file
        """
        pass
