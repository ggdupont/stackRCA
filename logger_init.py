import logging


def init_logger(name):
    the_logger = logging.getLogger(name)
    the_logger.setLevel(logging.INFO)
    if not len(the_logger.handlers):
        fh = logging.FileHandler('./logs/{}.log'.format(name))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add the handlers to the logger
        the_logger.addHandler(fh)
        the_logger.addHandler(ch)
        
    return the_logger