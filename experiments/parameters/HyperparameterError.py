class HyperparameterError(Exception):
    """
    An exception that is thrown when an error with hyperparameters occurred.
    """

    def __init__(self, message: str):
        """
        Initializes the exception.
        :param message: The error message.
        """
        super().__init__(message)


class HyperparameterFileError(HyperparameterError):
    """
    An exception that is thrown when an error with the hyperparameter file occurred.
    """

    def __init__(self, message: str, param_file: str):
        """
        Initializes the exception.
        :param message: The error message.
        :param param_file: The parameter file.
        """
        super().__init__(message)
        self.file = param_file