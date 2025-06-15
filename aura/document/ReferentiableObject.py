from uuid import uuid4


class ReferentiableObject:
    def __init__(self, id_ = None):
        self._id = id_

    @property
    def id(self):
        if self._id is None:
            self._id = str(uuid4())

        return self._id
