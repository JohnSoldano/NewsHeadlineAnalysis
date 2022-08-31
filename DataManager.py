import GetData as GetData
import numpy as np
import pandas as pd
import sqlite3
sqlite3.register_adapter(np.int64, lambda val: int(val))
sqlite3.register_adapter(np.int32, lambda val: int(val))

class DataManager:
    def __init__(self, rics):
        # string: name of object?
        self.name = rics
        # string: file location of rics
        self.rootDirectory = "D:/MyProjects/NewsHeadlineAnalysis/Data/"
        
        # objects for each data
        self.hl = GetData('Headlines', self.rootDirectory, self.name)
        self.ts = GetData('Timeseries', self.rootDirectory, self.name)
        
        # Creates dataframe from headlines and timeseries data
        self.df_ts, self.df_hl = self.GetDataFrames()
        
        # Converts into single dataframe for machine learning
        self.df = self.CreateDataFrame()
    
    def CreateDataFrame(self):
        # Create mmapping of ts dates to index
        ts_dict = {i: self.df_ts[self.df_ts['Date'] == i].index for i in self.df_ts['Date'].unique()}
        hl_index_dict = self.IndexHL2TS(ts_dict)
        hl_data = self.CreateDataDictionary_HL(ts_dict, hl_index_dict)
        df_values = self.CreateDataFrameValues(hl_data)
        df = self.CreateFinalDataFrame(df_values, hl_data)
        return df
    
    def CreateFinalDataFrame(self, df_values, hl_data):
        col_seq = [*reversed(range(1,31))]
        ts_cols = ['t-'+str(j) for j in col_seq]
        num_cols = ['HIGH','LOW','OPEN','CLOSE','COUNT','VOLUME']
        all_cols = ['DateTimeBLOB','DateTime', 'text'] + ts_cols + num_cols + ['30M_CLOSE']
        df = pd.DataFrame(df_values,columns=all_cols, index=hl_data.keys())
        return df
    
    def CreateDataFrameValues(self, hl_data):
        vals = []
        for i in hl_data.keys():
            A = np.array(self.df_hl.loc[i][['DateTimeBLOB','DateTime','text']].values).reshape(1,-1)
            B = np.array(self.df_ts.loc[hl_data[i]['30M_BEFORE_index']]['CLOSE'].values).reshape(1,-1)
            C = np.array(self.df_ts.loc[hl_data[i]['index_at_hl']][['HIGH','LOW','OPEN','CLOSE','COUNT','VOLUME']].values).reshape(1,-1)
            D = hl_data[i]['CLOSE_30M']
            vals += [np.concatenate((A,B,C,D),axis=None)]
        return vals
        
    
    def CreateDataDictionary_HL(self, ts_dict, hl_index_dict):
        hl_data = dict()
        for i in hl_index_dict.keys():
            if (self.check_index_range(self.df_hl, i, ts_dict, hl_index_dict)):
                hl_data[i] = {
                    '30M_BEFORE_index': [hl_index_dict[i]-j for j in reversed(range(1,31))],
                    'index_at_hl': hl_index_dict[i],
                    'CLOSE_30M': self.df_ts.loc[hl_index_dict[i]+30]['CLOSE']
                    }
        return hl_data
    
    def IndexHL2TS(self, ts_dict):
        hl_dict = dict()
        for i in self.df_hl.index:
            temp = self.df_ts.loc[ts_dict[self.df_hl.loc[i]['Date']]]
            
            # if statement was added because headline datetime and timeseries datetime are not one-to-one
            if (self.df_hl['DateTime'][i] in [*temp['DateTime']]):
                hl_dict[i] = temp[temp['DateTime'] == self.df_hl['DateTime'][i]].index[0]
        return hl_dict
        
    def LoadTwoDatabases(self, db1, db2):
        db1_fn = db1.dirFilePath
        db2_fn = db2.dirFilePath
        conn = sqlite3.connect(db1_fn)
        cur = conn.cursor()
        x = ''' ATTACH DATABASE '{}' AS db2'''.format(db2_fn).replace("'",'"')
        conn.execute(x)
        query = '''SELECT * FROM "{}" 
        WHERE Date IN (SELECT DISTINCT Date FROM "{}" INTERSECT SELECT DISTINCT Date FROM db2."{}") AND 
        Time >= '09:33:00' AND 
        Time <= '15:30:00' AND
        strftime('%w', Date) != '6' AND
        strftime('%w', Date) != '0'
        '''.format(self.name, self.name, self.name).replace('"','')
        cur.execute(query)
        vals = [i for i in cur.fetchall()]
        db_df = pd.DataFrame(vals, columns=db1.colNames)
        db_df.set_index('id', drop=True, inplace=True)
        return(db_df)
    
    def GetDataFrames(self):
        ts = self.LoadTwoDatabases(self.ts, self.hl)
        hl = self.LoadTwoDatabases(self.hl, self.ts)
        return [ts, hl]

# return: Bool,
# function: determines existstance of 30 indexes before and after news headline time
    def check_index_range(self, hl, hl_index, ts_dict, hl_index_dict):
        CLOSE_30M_AFTER = False
        CLOSE_30M_BEFORE = False
        ts_index = ts_dict[hl.loc[hl_index]['Date']]
        
        if (max(ts_index) - hl_index_dict[hl_index] >= 30):
            CLOSE_30M_AFTER = True
        if (hl_index_dict[hl_index] - min(ts_index) >= 30):
            CLOSE_30M_BEFORE = True
            
        if ((CLOSE_30M_AFTER == True) and (CLOSE_30M_BEFORE == True)):
            return True
        else:
            return False