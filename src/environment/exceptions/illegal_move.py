class IllegalMoveException(Exception):
    def __init__(self):
        self.message = "Move is not legal"
        super().__init__(self.message)
