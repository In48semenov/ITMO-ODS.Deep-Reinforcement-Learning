from typing import Dict

from clearml import Task

import os
from dotenv import load_dotenv
load_dotenv("./clearml.env")

web_server = os.getenv("web_server")
api_server = os.getenv("api_server")
files_server = os.getenv("files_server")
access_key = os.getenv("access_key")
secret_key = os.getenv("secret_key")

Task.set_credentials(
    web_host=web_server,
    api_host=api_server,
    files_host=files_server,
    key=access_key,
    secret=secret_key
)


class Manager:

    def __init__(self, project: str, task_name: str):
        self.task = Task.init(
            project_name=project, task_name=task_name,
        )

        self.logger = self.task.get_logger()

    def log_params(self, params: Dict) -> None:
        self.task.connect(params)

    def report_metrics(
            self, title: str, series: str, loss: float, iteration: int
    ) -> None:
        self.logger.report_scalar(
            title=title,
            series=series,
            value=loss,
            iteration=iteration
        )

    def close_task(self, ):
        self.task.close()
