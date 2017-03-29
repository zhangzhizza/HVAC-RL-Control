import pandas as pd;
import numpy as np;

def get_time_interpolate(dataframe, time):
    """
    The function returns row corresponding to the index defined by time.
    If not exist, then interpolated data will be returned. If interpolation
    returns nan which occurs when interpolating beyond the first row of the
    data, then the first row of the data will be returned (e.g. The olddest
    data in the dataframe is 1991-02-01 00:01:00, when requested for data at
    1991-02-01 00:00:00, the 1991-02-01 00:01:00 row will be returned). If 
    interpolation beyond the last row of the data, then the first row of the 
    data will be returned. 
    
    Args:
        dataframe: pd.DataFrame
            DataFrame object with time as its index. DataFrame must sorted
            by the time index from old to new.
        time: String
            Time represented as the String.
    
    Return: 1-D python list.
        The interpolated values at the time. 
    """
    # If the time index exists, just return that index row
    if time in dataframe.index:
        return dataframe.loc[time].as_matrix().flatten();
    
    # Else, do interpolation
    indexAfterTime = (dataframe.index >= time).argmax();
    relevantRows = dataframe.iloc[indexAfterTime - 1:indexAfterTime + 1];
    relevantRows_inserted = relevantRows.reindex(pd
                                                 .to_datetime(
                                                     list(relevantRows.index.values) + 
                                                     [pd.to_datetime(time)]
                                                     )
                                                 );
                                                 
    relevantRows_interpolated = relevantRows_inserted.interpolate('time') \
                                                     .loc[time];
    
    ret = relevantRows_interpolated.as_matrix().flatten();
    
    # Check if exist nan, usually occurs when interpolation out of the first
    # entry of the data
    for item in ret:
        if np.isnan(item):
            ret = dataframe.iloc[indexAfterTime].as_matrix().flatten();
            break;
    
    return ret.tolist();
    
    
    
