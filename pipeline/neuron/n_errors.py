class UnexpectedMutationError(Exception):
    """
    Error class for incomplete or invalid HGVS-style mutation codes
    """

    pass


class UnexpectedResidueError(Exception):
    """
    Error class for invalid sequences
    """

    pass
