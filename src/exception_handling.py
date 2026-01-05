import sys
import logging

def get_error_details(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()

    line_number = exc_tb.tb_lineno
    file_name = exc_tb.tb_frame.f_code.co_filename
   
    error_message = str(error)
    detailed_message = f"Error occurred in script: {file_name} at line number: {line_number}. Error message: {error_message}"

    return detailed_message


class CustomException(Exception):

    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = get_error_details(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 10/0
    except Exception as e:
        logging.info("divide by Zero error")
        raise CustomException(e, sys)