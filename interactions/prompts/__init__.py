# Decorator that catches exceptions to print the response for debugging and re-raises the exception
def response_debug(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            response = args[0] if args else kwargs.get('response', '')
            print("Debugging Response:\n", response)
            raise e
    return wrapper