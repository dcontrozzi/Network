import pandas as pd

class ETF:

    def __init__(self, etfName):
        self.etfName = etfName

    def loadConstituents(self, path):
        self.etfConstituents = pd.read_csv(path)
