from abc import ABC, abstractmethod


class IInfographicsParser(ABC):
    """Interface for creating the infographic"""

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
