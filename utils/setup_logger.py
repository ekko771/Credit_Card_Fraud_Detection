import os
import logging
import logging.config
import yaml
from logging.handlers import RotatingFileHandler

with open("{}/{}".format(os.path.dirname(__file__), "log.yaml"), 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
logger = logging.getLogger('app')

'''
import datetime
log_filename = datetime.datetime.now().strftime("./log/tk%Y-%m-%d_%H_%M_%S.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    filename=log_filename)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logger = logging.getLogger('app')
'''
