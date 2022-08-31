from DataManager import GetData
from DataManager import DataManager

class Company(DataManager):
    def __init__(self, rics):
        # Type: string. ASX code for company
        self._name = rics + "_AX"
        DataManager.__init__(self, rics)