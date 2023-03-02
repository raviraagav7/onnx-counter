import abc
import logging
from pathlib import Path
from turtle import Screen
from rich.table import Table
from rich import box

class DisplaySummary(metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        self.table = None
        self.table_name = None
        self.display_logger = logging.getLogger(f'onnx_profiling.onnx_profile.{str(Path(__file__).stem)}.DisplaySummary')
    
    @abc.abstractmethod
    def _set_header(self) -> None:
        raise NotImplementedError

    def _create_table(self):
        """Make a new table."""
        self.table = Table(title=f'[bold bright_green]Model : {self.table_name}', box=box.DOUBLE_EDGE, title_justify='center')
        self._set_header()

    @abc.abstractmethod
    def generate_table(self, table_name: str):
        raise NotImplementedError