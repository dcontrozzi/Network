from enum import Enum


class TenorBucket(Enum):
    T0YR = 0
    T2YR = 2
    T3YR = 3
    T5YR = 5
    T10YR = 10
    T30YR = 30

    @staticmethod
    def from_years_to_maturity(ytm):
        if ytm < 0.5:
            return TenorBucket.T0YR
        if ytm < 1.5:
            return TenorBucket.T2YR
        elif ytm < 3.5:
            return TenorBucket.T3YR
        elif ytm < 6.5:
            return TenorBucket.T5YR
        elif ytm < 15:
            return TenorBucket.T10YR
        else:
            return TenorBucket.T30YR