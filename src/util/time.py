"""Time utility functions"""

import datetime

def get_time_from_seconds(second, start_year, start_mon, 
                          start_date, start_day):
    """
    Get the day of the week and hour of the day given the delta seconds and
    the start time.
    
    Args:
        second: int
            Delta seconds from the start time.
        start_year: int 
            Start year.
        start_mon: int 
            Start month.
        start_date: int.
            The day of the month at the start time.
        start_day: int 
            The start weekday. 0 is Monday and 6 is Sunday.
    
    Return: (int, int)
        The hour of the day and the weekday of now (hour of the day ranges from
        0 to 23, and weekday ranges from 0 to 6 where 0 is Monday).
    """
    startTime = datetime.datetime(start_year, start_mon, start_date, 0, 0, 0);
    nowTime = startTime + datetime.timedelta(0, second);
    delta_days = (nowTime - startTime).days;
    nowWeekday = (start_day + delta_days) % 7;
    nowHour = nowTime.hour;
    return (nowWeekday, nowHour);
