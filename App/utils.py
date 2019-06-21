import logging
import os
from datetime import datetime

class twitter_model_logging(object):
    
    def __init__(self):
        
        logging.basicConfig(filename='twitter_model.log',level=logging.critical)

    def check_file_path(self,file):

        if os.path.exists(file) == False:

            logging.critical(file + ' not found. ' + datetime.now().strftime('%a %b %d %Y at %H:%M:%S'))

            if os.path.exists('/backups/' + file) == False:

                logging.critical('Backup file not found. Raised IOError.' + datetime.now().strftime('%a %b %d %Y at %H:%M:%S'))

                raise IOError('Internal error.')
                
        return file

    def check_array_shape(self,array,dimensions):

        logging.basicConfig(filename='twitter_model.log',level=logging.DEBUG)
        check = array.shape[1] == dimensions

        if check == False:

            logging.critical('Incorrect embedded train vec shape. ' + str(self.train_embedded_vecs.shape[1]) + ' != ' + \
                             str(dimension) + '. Raised ValueError. ' + datetime.now().strftime('%a %b %d %Y at %H:%M:%S'))
            raise ValueError('Internal error.')