

class BondIndic:
    def __init__(self,
                 isin = None,
                 id_cusip = None,
                 issue_dt = None,
                 security_des = None,
                 ticker = None,
                 cpn = None,
                 cpn_typ = None,
                 maturity = None,
                 mty_years = None,
                 series = None,
                 is_144_A = None,
                 is_reg_s = None,
                 industry_sector = None,
                 industry_group = None,
                 mty_typ = None,
                 calc_typ = None,
                 calc_typ_des = None,
                 country = None,
                 currency = None,
                 amt_issued = None,
                 amt_outstanding = None,
                 callable = None,
                 sinkable = None,
                 is_perpetual = None,
                 putable = None,
                 floater = None,
                 day_cnt_des = None,
                 bond_to_eqy_ticker = None,
                 payment_rank = None,
                 bb_composite = None,
                 rtg_sp = None,
                 rtg_moody = None,
                 rtg_fitch = None,
                 is_trace_eligible = None,
                 is_subordinated = None,
                 is_fix_to_float = None,
                 is_secured = None,
                 is_covered = None,
                 blp_spread_benchmark_name = None,
                 nasd_trace_benchmark = None,
                 yas_bnchmrk_bond = None):
        self.isin = isin
        self.id_cusip = id_cusip
        self.issue_dt = issue_dt
        self.security_des = security_des
        self.ticker = ticker
        self.cpn = cpn
        self.cpn_typ = cpn_typ
        self.maturity = maturity
        self.mty_years = mty_years
        self.series = series
        self.is_144_A = is_144_A
        self.is_reg_s = is_reg_s
        self.industry_sector = industry_sector
        self.industry_group = industry_group
        self.mty_typ = mty_typ
        self.calc_typ = calc_typ
        self.calc_typ_des = calc_typ_des
        self.country = country
        self.currency = currency
        self.amt_issued = amt_issued
        self.amt_outstanding = amt_outstanding
        self.callable = callable
        self.sinkable = sinkable
        self.is_perpetual = is_perpetual
        self.putable = putable
        self.floater = floater
        self.day_cnt_des = day_cnt_des
        self.bond_to_eqy_ticker = bond_to_eqy_ticker
        self.payment_rank = payment_rank
        self.bb_composite = bb_composite
        self.rtg_sp = rtg_sp
        self.rtg_moody = rtg_moody
        self.rtg_fitch = rtg_fitch
        self.calc_typ_des = calc_typ_des
        self.is_trace_eligible = is_trace_eligible
        self.is_subordinated = is_subordinated
        self.is_fix_to_float = is_fix_to_float
        self.is_secured = is_secured
        self.is_covered = is_covered
        self.blp_spread_benchmark_name = blp_spread_benchmark_name
        self.nasd_trace_benchmark = nasd_trace_benchmark
        self.yas_bnchmrk_bond = yas_bnchmrk_bond


    def get_rating(self):
        '''
        simple rating aggregator
        :return:
        '''
        if self.bb_composite:
            return self.bb_composite
        elif self.rtg_sp:
            return self.rtg_sp
        elif self.rtg_moody:
            return self.rtg_moody
        elif self.rtg_fitch:
            return self.rtg_fitch
        else:
            return None