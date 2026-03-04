class ApiError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class ValidationError(ApiError):
    def __init__(self, message: str):
        super().__init__(400, message)
