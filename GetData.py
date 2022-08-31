import eikon
eikon.set_app_key('##########################################') # Private KEY

import sqlite3
import numpy as np
import pickle
import pandas as pd
from dateutil.tz import tzlocal
from dateutil import tz
from datetime import datetime, timedelta
from dateutil.rrule import rrule, MONTHLY
from calendar import monthrange
import pytz
sqlite3.register_adapter(np.int64, lambda val: int(val))
sqlite3.register_adapter(np.int32, lambda val: int(val))

class GetData(object):
    def __init__(self, type, dir, rics):
        self.name = rics    # string: RICS
        self.type = type    # string: headlines/timeseries
        self.dirRoot = dir  # string: root directory
        self.dirPath = self.dirRoot + type + "/"    # string: root directory + database folder
        self.dirFilePath = self.dirPath + self.type + '.db' # string: complete pathway to database
        
        self.colNames, self.dataType = self.GetColNameDatatype(self.type) # list<string>: colNames/dataType for tables & dataframes
        self.finalDateTime = 0
        
        # Automatically creates/updates data upon creation
        self.Main()

    # Main function of GetData
    def Main(self):
        # 1)    Checks if a table exists within database
        #       creates table and downloads data
        # bool  :   true    > database doesn't exist needs to be created.
        #       :   false   > database exists already
        if (self.TableExists() == False):
            self.CreateTable()
        
        # 2)    Checks if table is up-to-date (else data is already up-to-date)
        # bool  :   true    > database isn't up-to-date and needs to be updated.
        #       :   false   > database is up-to-date.
        if (self.UpdateTable() == True):
            data = self.DownloadData()
            self.UpdateDatabase(data)
            
    def UpdateTable(self):
        current_datetime = datetime.today().replace(microsecond=0, tzinfo=tzlocal())
        if (self.GetLastTableEntry() < current_datetime):
            return True
        return False
    
    # Function to create table within database for RICS
    def CreateTable(self):
        print('initialising ', self.type, ' table for ', self.name, sep='')
        # create string to parse to cursor
        str_to_parse = '''CREATE TABLE "{}"('''.format(self.name).replace('"','')
        for i in range(0, len(self.colNames)):
            str_to_parse += '''"{}" "{}", '''.format(
                    self.colNames[i],
                    self.dataType[i]).replace('"','')
        str_to_parse = str_to_parse[:-2] + ''')'''
        
        # load database/cursor/create new table
        db = sqlite3.connect(self.dirFilePath)
        cur = db.cursor()
        cur.execute(str_to_parse)
        
        # set table exist to true
        self.tableExists = True
        
    # check if table exists. return true or false
    def TableExists(self):
        db = sqlite3.connect(self.dirFilePath)
        cur = db.cursor()
        # Extract final entry from table
        cur.execute('''SELECT count(name) FROM sqlite_master WHERE type='table'
                    AND name ='"{}"' '''.format(self.name).replace('"',''))
        if cur.fetchone()[0] == 1:
            return True
        else:
            print(self.type,' ==> Table: ', self.name, ' does not exist.',sep='')
        return False

    # Download the required data and save to database?
    def DownloadData(self):
        # Generates start and ending dates to download data from
        [start, end] = self.GenerateDates()
        
        # Download new data
        data = self.GetData(start, end)
        
        # Format then append data to database
        data = self.FormatDateTimeColumns(data)
        return data
    
    # Format Date Time Columns of newly downloaded data to match the existing database
    def FormatDateTimeColumns(self, data):
        # create tz aware for news headlines
        if (self.type == "Headlines"):
            data['DateTime'] = data['DateTime'].apply(lambda x: x.astimezone(tz.gettz('Australia/Adelaide')).replace(microsecond=0))
            data['DateTime'] = data['DateTime'].apply(lambda x: self.round_headline_time(x))
            # 
        elif (self.type == "Timeseries"):
            # convert set datetime to tzaware
            data['DateTime'] = data['DateTime'].apply(lambda x: x.tz_localize(pytz.timezone('GMT')).astimezone(tz.gettz('Australia/Adelaide')))
        
        ## Split DateTime into datetime (this is done for selecting range)
        data[['Date','Time']] = [(x.date(), x.time()) for x in data['DateTime']]
        data.insert(0,'DateTimeBLOB', data['DateTime'].apply(lambda x: pickle.dumps(x)))
        return data
    
    def UpdateDatabase(self, data):
        # convert datetime to string (requirement for sql)
        columns_to_string = self.colNames[-3:]
        for i in columns_to_string:
            data[i] = data[i].apply(lambda x: str(x))

        # save data as tuple
        df_to_tuple_list = []
        for i in data.index:
            df_to_tuple_list += [tuple(data.iloc[i, data.columns.get_loc(j)] for j in data.columns)]

        # prepare to save to sql
        a='''INSERT INTO "{}"(''' + len(data.columns)*'''"{}",'''
        b=a[:-1] + ''') VALUES(''' + len(data.columns)*'''?,'''
        c=b[:-1] + ''')'''
        d = [self.name] + self.colNames[1:]
        e = c.format(*d).replace('"','')
        db = sqlite3.connect(self.dirFilePath)
        cur = db.cursor()
        cur.executemany(e, df_to_tuple_list)
        db.commit()
    
    # Downloads data for the given data type and passed start/end date time
    def GetData(self, start, end):
        # downloads and appends headlines for the given time date intervals
        if (self.type == "Headlines"):
            hl_data = pd.DataFrame(columns = self.colNames[2:-2])
            it = 1
            if (isinstance(start, list)):
                num_downloads = len(start)
            else:
                num_downloads = 1
            
            print('connecting to refinitiv...')
            for i,j in zip(start, end):
                print('downloading: ', it ,'/', num_downloads, sep='')
                it += 1
                temp_df = eikon.get_news_headlines(query = 'R:' + self.name.replace('_','.'),
                                                date_from = i,
                                                date_to = j,
                                                count = 100)
                if len(temp_df) > 0:
                    temp_df = temp_df.reset_index(drop=True)
                    temp_df = temp_df.rename(columns={'versionCreated': 'DateTime'})
                    temp_df = temp_df[hl_data.columns].sort_values('DateTime')
                    hl_data = hl_data.append(temp_df)
            
            hl_data = hl_data.reset_index(drop=True)
            return hl_data
        elif (self.type == "Timeseries"):
            ts_data = pd.DataFrame(columns = self.colNames[2:-2])
            it = 1
            num_downloads = len(start)
            
            print('connecting to refinitiv...')
            for i,j in zip(start, end):
                print('downloading: ', it ,'/', num_downloads, sep='')
                it += 1
                temp_df = eikon.get_timeseries(
                    rics = self.name.replace('_','.'),
                    start_date = i,
                    end_date = j,
                    interval='minute'
                )
                if (temp_df is not None):
                    temp_datetime = temp_df.index
                    temp_df = temp_df.reset_index(drop=True)
                    temp_df['DateTime'] = temp_datetime
                    ts_data = ts_data.append(temp_df)
            
            ts_data = ts_data.reset_index(drop=True)
            return ts_data
    
    # Generates Datetime for given type
    def GenerateDates(self):
        start = self.finalDateTime
        end = datetime.today().replace(microsecond=0, tzinfo=tzlocal())
        if (self.type == 'Headlines'):            
            # create sequence for dates to iterate through
            s = [start]
            e = []
            
            while (s[-1] < end):
                s += [s[-1] + timedelta(days=2)]
                e += [s[-1] + timedelta(seconds=-1)]
            
            s = s[:-1]
            if e[-1] > end:
                e[-1] = end
            return s,e
        
        elif (self.type == 'Timeseries'):
            s = [x for x in rrule(MONTHLY, dtstart=start-timedelta(days=start.day-1), until=end)]
            e = [x + timedelta(days=monthrange(x.year, x.month)[1]-1, hours=23, minutes=59, seconds=59) for x in s]
            #
            s[0] = start
            e[-1] = end
            return s, e

    # gets last datetime from database
    def GetLastTableEntry(self):
        db = sqlite3.connect(self.dirFilePath)
        cur = db.cursor()
        cur.execute('''SELECT * FROM "{}" ORDER BY id DESC LIMIT 1'''.format(self.name).replace('"',''))
        last_row = cur.fetchall()[0]
        self.finalDateTime = pickle.loads(last_row[1]).to_pydatetime()
        return self.finalDateTime

    # Dictionary containing column names and data type for each database
    #   needed for initial creation of table
    def GetColNameDatatype(self, type):
        data_dict = dict()
        data_dict['Headlines'] = {'column_names': ['id',                     'DateTimeBLOB', 'text', 'storyId', 'sourceCode',    'DateTime','Date','Time'],
                         'column_datatype': ['INTEGER PRIMARY KEY', 'BLOB',         'TEXT', 'TEXT',     'TEXT',         'TEXT','    TEXT','TEXT']}

        data_dict['Timeseries'] = {'column_names':       ['id',                  'DateTimeBLOB',     'HIGH', 'LOW',  'OPEN', 'CLOSE', 'COUNT',   'VOLUME',   'DateTime', 'Date', 'Time'],
                                'column_datatype':    ['INTEGER PRIMARY KEY', 'BLOB',             'REAL', 'REAL', 'REAL', 'REAL',  'INTEGER', 'INTEGER',  'TEXT',     'TEXT', 'TEXT']}

        data_dict['NewsText'] = {'column_names': ['id','storyId','storyText'],
                                'column_datatype': ['INTEGER', 'TEXT', 'TEXT']}

        return data_dict[type]['column_names'], data_dict[type]['column_datatype']
    
    
    # rounds seconds UP the nearest minute for headlines
    def round_headline_time(self, df):
        if (df.second != 0):
            df = df.replace(second=0)
            df = df + timedelta(minutes=1)
        return df  
