from enum import Enum

class Sectors(Enum):
    Financial = 1
    BasicMaterials = 2
    Industrial = 3
    ConsumerCyclical = 4
    Utilities = 5
    Technology = 6
    Energy = 7
    Communications = 8
    ConsumerNonCyclical = 9
    INVALID = 10

    @staticmethod
    def is_valid(sector_string):
        return Sectors.from_string(sector_string) in list(Sectors)

    @staticmethod
    def from_string(sector_string):
        if sector_string == 'Financial':
            return Sectors.Financial
        elif sector_string == 'Basic Materials':
            return Sectors.BasicMaterials
        elif sector_string == 'Industrial':
            return Sectors.Industrial
        elif sector_string == 'Consumer, Cyclical':
            return Sectors.ConsumerCyclical
        elif sector_string == 'Utilities':
            return Sectors.Utilities
        elif sector_string == 'Technology':
            return Sectors.Technology
        elif sector_string == 'Energy':
            return Sectors.Energy
        elif sector_string == 'Communications':
            return Sectors.Communications
        elif sector_string == 'Consumer, Non-cyclical':
            return Sectors.ConsumerNonCyclical
        else:
            return Sectors.INVALID

    @staticmethod
    def to_string(sector):
        if sector == Sectors.BasicMaterials:
            return 'Basic Materials'
        elif sector == Sectors.ConsumerCyclical:
            return 'Consumer, Cyclical'
        elif sector == Sectors.ConsumerNonCyclical:
            return 'Consumer, Non-cyclical'
        else:
            return sector.name

