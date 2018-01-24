import logging
import os.path

class Logger():
    
    def getLogger(self, name, level, formatter, log_file_path = None):
        logger = logging.getLogger(name);
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logging.Formatter(formatter));
        logger.addHandler(consoleHandler);
        if log_file_path is not None:
            if not os.path.isfile(log_file_path):
                logfile = open(log_file_path, 'w+'); # Create a new log file
                logfile.close();
            fileHandler = logging.FileHandler(log_file_path)
            fileHandler.setFormatter(logging.Formatter(formatter))
            logger.addHandler(fileHandler);    
        logger.setLevel(level);
        logger.propagate = False;
        return logger;