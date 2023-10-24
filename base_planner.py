class BasePlanner:
    def __init__(self, **kwargs):
        """Set up hyper-parameters."""

        self.path = []

    def plan(self, map, start_conf, end_conf):
        """path planning from start to end in the given map."""
        raise NotImplementedError("Method undefined...")
