class NoMovesException(Exception):
    def __init__(self):
        self.message = "No legal moved could be found"
        super().__init__(self.message)
