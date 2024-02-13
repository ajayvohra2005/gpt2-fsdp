import logging

logging.basicConfig(
            format='%(levelname)s:%(process)d:%(message)s',
            level=logging.INFO)

def get_logger():
    return logging.getLogger("fsdp-gp2")