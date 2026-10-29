class CheckError(Exception):
    """A program the checker refuses. `problems` lists every problem found, each with its line."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))
