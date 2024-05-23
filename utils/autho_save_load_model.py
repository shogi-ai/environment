"""A decorator function for saving and loading a model before and after calling the decorated function."""


def save_load_model(func):
    """
    A decorator function for saving and loading a model before and after calling the decorated function.

    Parameters:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function that loads the model, calls the decorated function, and then saves the model.

        Parameters:
            *args: Positional arguments to be passed to the decorated function.
            **kwargs: Keyword arguments to be passed to the decorated function.

        Returns:
            Any: The return value of the decorated function.
        """
        model_path = "./models/models_base.pth"
        args[0].get_model(model_path)
        result = func(*args, **kwargs)
        args[0].save_model(model_path)

        return result

    return wrapper
