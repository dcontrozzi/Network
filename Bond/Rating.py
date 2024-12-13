from enum import Enum

class AggregatedRating(Enum):
    """
    Aggregate AAA, AA, AA+, AA- into HIGH_A
    """
    HIGH_A = 1
    APlus = 2
    A = 3
    AMinus = 4
    BBBPlus = 5
    BBB = 6
    BBBMinus = 7
    NOTIG = 8

    @staticmethod
    def from_str(rating):
        if rating in ('AAA', 'AA+', 'AA', 'AA-') :
            return AggregatedRating.HIGH_A
        elif rating == 'A+':
            return AggregatedRating.APlus
        elif rating == 'A':
            return AggregatedRating.A
        elif rating == 'A-':
            return AggregatedRating.AMinus
        elif rating == 'BBB+':
            return AggregatedRating.BBBPlus
        elif rating == 'BBB':
            return AggregatedRating.BBB
        elif rating == 'BBB-':
            return AggregatedRating.BBBMinus
        else:
            return AggregatedRating.NOTIG

    @staticmethod
    def to_str(rating):
        if rating == AggregatedRating.HIGH_A:
            return 'HIGH_A'
        elif rating == AggregatedRating.APlus:
            return 'A+'
        elif rating == AggregatedRating.A:
            return 'A'
        elif rating == AggregatedRating.AMinus:
            return 'A-'
        elif rating == AggregatedRating.BBBPlus:
            return 'BBB+'
        elif rating == AggregatedRating.BBB:
            return 'BBB'
        elif rating == AggregatedRating.BBBMinus:
            return 'BBB-'
        else:
            return 'NOTIG'

    @staticmethod
    def from_rating(rating):
        if rating in (Rating.AAA, Rating.AAPlus, Rating.AA, Rating.AAMinus):
            return AggregatedRating.HIGH_A
        elif rating == Rating.APlus:
            return AggregatedRating.APlus
        elif rating == Rating.A:
            return AggregatedRating.A
        elif rating == Rating.AMinus:
            return AggregatedRating.AMinus
        elif rating == Rating.BBBPlus:
            return AggregatedRating.BBBPlus
        elif rating == Rating.BBB:
            return AggregatedRating.BBB
        elif rating == Rating.BBBMinus:
            return AggregatedRating.BBBMinus
        else:
            return AggregatedRating.NOTIG

class RatingCategory(Enum):
    HIGH = 1
    LOW = 2

    @staticmethod
    def get_rating_category(rating):
        return RatingCategory.LOW if rating in (Rating.BBB, Rating.BBBPlus, Rating.BBBMinus, Rating.NOTIG) \
            else RatingCategory.HIGH

class Rating(Enum):
    AAA = 1
    AAPlus = 2
    AA = 3
    AAMinus = 4
    APlus = 5
    A = 6
    AMinus = 7
    BBBPlus = 8
    BBB = 9
    BBBMinus = 10
    NOTIG = 11


    @staticmethod
    def from_str(rating):
        if rating == 'AAA':
            return Rating.AAA
        elif rating == 'AA+':
            return Rating.AAPlus
        elif rating == 'AA':
            return Rating.AA
        elif rating == 'AA-':
            return Rating.AAMinus
        elif rating == 'A+':
            return Rating.APlus
        elif rating == 'A':
            return Rating.A
        elif rating == 'A-':
            return Rating.AMinus
        elif rating == 'BBB+':
            return Rating.BBBPlus
        elif rating == 'BBB':
            return Rating.BBB
        elif rating == 'BBB-':
            return Rating.BBBMinus
        else:
            return Rating.NOTIG

    @staticmethod
    def to_str(rating):
        if rating == Rating.AAA:
            return 'AAA'
        elif rating == Rating.AAPlus:
            return 'AA+'
        elif rating == Rating.AA:
            return 'AA'
        elif rating == Rating.AAMinus:
            return 'AA-'
        elif rating == Rating.APlus:
            return 'A+'
        elif rating == Rating.A:
            return 'A'
        elif rating == Rating.AMinus:
            return 'A-'
        elif rating == Rating.BBBPlus:
            return 'BBB+'
        elif rating == Rating.BBB:
            return 'BBB'
        elif rating == Rating.BBBMinus:
            return 'BBB-'
        else:
            return 'NOTIG'

