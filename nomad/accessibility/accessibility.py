class Accessibility():
    def __init__(self, od_matrix, mode_subset, access_type):
        '''Description here.'''
        self.od_matrix = od_matrix
        self.mode_subset = mode_subset
        self.access_type = access_type

    def calc_cumulative_opps(self, mode_subset, access_type):
        '''Requires that the od matrix: 1) has been converted to index form; 2) has been merged with df_dst_opps.'''

        if self.access_type not in ["time", "money", "both"]:
            raise ValueError("Invalid value for 'access_type'. Expected 'time', 'money', or 'both'.")

        od_matrix_subset = self.od_matrix.loc[self.od_matrix['mode_subset'] == mode_subset]

        if self.access_type == "time":
            od_matrix_subset['opp_total_time'] = self.od_matrix[self.od_matrix.opp_col_name] * self.od_matrix['access_indicator_time']
            self.cumulative_opps = od_matrix_subset[['org_idx', 'opp_total_time']].groupby('org_idx')['opp_total_time'].sum().reset_index().sort_values(by='opp_total_time', ascending=True)
        elif self.access_type == "money":
            od_matrix_subset['opp_total_money'] = self.od_matrix[self.od_matrix.opp_col_name] * self.od_matrix['access_indicator_money']
            self.cumulative_opps = od_matrix_subset[['org_idx', 'opp_total_money']].groupby('org_idx')['opp_total_money'].sum().reset_index().sort_values(by='opp_total_money', ascending=True)
        elif self.access_type == "both":
            od_matrix_subset['opp_total_time'] = self.od_matrix[self.od_matrix.opp_col_name] * self.od_matrix['access_indicator_time']
            self.cumulative_opps = od_matrix_subset[['org_idx', 'opp_total_time']].groupby('org_idx')['opp_total_time'].sum().reset_index().sort_values(by='opp_total_time', ascending=True)
            od_matrix_subset['opp_total_money'] = self.od_matrix[self.od_matrix.opp_col_name] * self.od_matrix['access_indicator_money']
            self.cumulative_opps = od_matrix_subset[['org_idx', 'opp_total_money']].groupby('org_idx')['opp_total_money'].sum().reset_index().sort_values(by='opp_total_money', ascending=True)
            od_matrix_subset['opp_total_time_money'] = self.od_matrix[self.od_matrix.opp_col_name] * self.od_matrix['access_indicator_time'] * self.od_matrix['access_indicator_money']
            self.cumulative_opps = od_matrix_subset[['org_idx', 'opp_total_time_money']].groupby('org_idx')['opp_total_time_money'].sum().reset_index().sort_values(by='opp_total_time_money', ascending=True)

        

    @classmethod
    def calculate_accessibility(cls, od_matrix, mode_subset, access_type):
        # Initialize the calculator
        access = cls(od_matrix, mode_subset, access_type)
        access.calc_cumulative_opps()
        return access