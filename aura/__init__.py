from logging import getLogger, StreamHandler, INFO, DEBUG, Formatter

root = getLogger()

root.setLevel(DEBUG)

handler = StreamHandler()

formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

root.addHandler(handler)
