
class OD_matrix():
    def __init__(self, od_matrix, df_dst_opps, opp_col_name, travel_time_threshold, monetary_threshold):
        '''Description here.'''
        self.od_matrix = od_matrix
        self.opportunities = df_dst_opps
        self.opp_col_name = opp_col_name
        self.travel_time_threshold = travel_time_threshold
        self.monetary_threshold = monetary_threshold

    def get_indices(self):
        # Must be ordered for consistency
        orgs = sorted(set(self.od_matrix['org']))  
        dsts = sorted(set(self.od_matrix['dst']))
        self.org2idx = dict(zip(orgs, range(len(orgs))))
        self.dst2idx = dict(zip(dsts, range(len(dsts))))

    def add_od_indices(self):
        self.od_matrix['org_idx'] = self.od_matrix['org'].map(self.org2idx)
        self.od_matrix['dst_idx'] = self.od_matrix['dst'].map(self.dst2idx)

    def add_access_indicator(self):
        '''If no monetary threshold required, set arbitrarily high.'''

        self.od_matrix['access_indicator_time'] = (self.od_matrix['travel_time']/60 <= self.travel_time_threshold).astype(int) 
        self.od_matrix['access_indicator_monetary'] = (self.od_matrix['expense'] <= self.monetary_threshold).astype(int) 

    def merge_with_opps(self):
        self.opportunities['dst_idx'] = self.opportunities['dst'].map(self.dst2idx)
        self.od_matrix = self.od_matrix.merge(self.opportunities[['dst_idx', self.opp_col_name]], how='left', on='dst_idx')   

    @classmethod    
    def attach_data(cls, od_matrix, df_dst_opps, opp_col_name, travel_time_threshold, monetary_threshold):
        # Initialize the calculator
        od_matrix = cls(od_matrix, df_dst_opps, opp_col_name, travel_time_threshold, monetary_threshold)
        od_matrix.get_indices()
        od_matrix.add_od_indices()
        od_matrix.add_access_indicator()
        od_matrix.merge_with_opps()

        return od_matrix
    
    def calc_cumulative_opps(self, mode_subset, access_type):
        '''Requires that the od matrix: 1) has been converted to index form; 2) has been merged with df_dst_opps.'''

        if access_type not in ["time", "money", "both"]:
            raise ValueError("Invalid value for 'access_type'. Expected 'time', 'money', or 'both'.")

        od_matrix_subset = self.od_matrix.loc[self.od_matrix['mode_subset'] == mode_subset]

        if access_type == "time":
            od_matrix_subset['opp_total_time'] = self.od_matrix[self.opp_col_name] * self.od_matrix['access_indicator_time']
            cumulative_opps = od_matrix_subset[['org_idx', 'opp_total_time']].groupby('org_idx')['opp_total_time'].sum().reset_index().sort_values(by='opp_total_time', ascending=True)
            cumulative_opps.rename(columns = {'opp_total_time':'opportunities'}, inplace=True)
        
        elif access_type == "money":
            od_matrix_subset['opp_total_money'] = self.od_matrix[self.opp_col_name] * self.od_matrix['access_indicator_monetary']
            cumulative_opps = od_matrix_subset[['org_idx', 'opp_total_money']].groupby('org_idx')['opp_total_money'].sum().reset_index().sort_values(by='opp_total_money', ascending=True)
            cumulative_opps.rename(columns = {'opp_total_money':'opportunities'}, inplace=True)

        elif access_type == "both":
            # od_matrix_subset['opp_total_time'] = self.od_matrix[self.opp_col_name] * self.od_matrix['access_indicator_time']
            # cumulative_opps = od_matrix_subset[['org_idx', 'opp_total_time']].groupby('org_idx')['opp_total_time'].sum().reset_index().sort_values(by='opp_total_time', ascending=True)
            
            # od_matrix_subset['opp_total_money'] = self.od_matrix[self.opp_col_name] * self.od_matrix['access_indicator_monetary']
            # cumulative_opps = od_matrix_subset[['org_idx', 'opp_total_money']].groupby('org_idx')['opp_total_money'].sum().reset_index().sort_values(by='opp_total_money', ascending=True)
            
            od_matrix_subset['opp_total_time'] = self.od_matrix[self.opp_col_name] * self.od_matrix['access_indicator_time']
            od_matrix_subset['opp_total_money'] = self.od_matrix[self.opp_col_name] * self.od_matrix['access_indicator_monetary']
            od_matrix_subset['opp_total_time_money'] = self.od_matrix[self.opp_col_name] * self.od_matrix['access_indicator_time'] * self.od_matrix['access_indicator_monetary']
            cumulative_opps = od_matrix_subset[['org_idx', 'opp_total_time', 'opp_total_money', 'opp_total_time_money']].groupby('org_idx')['opp_total_time', 'opp_total_money', 'opp_total_time_money'].sum().reset_index().sort_values(by='opp_total_time_money', ascending=True)

            cumulative_opps.rename(columns = {'opp_total_time':'opportunities_time', 'opp_total_money':'opportunities_money', 'opp_total_time_money':'opportunities_time_money'}, inplace=True)
        
        return cumulative_opps

