from locust import HttpUser, TaskSet, task, between
import os 
import pandas as pd 
import random 
from loguru import logger


class Face_Analysis_Task(TaskSet):
    @task
    def predict(self):
        image_list = os.listdir("images")
        img_path=os.path.join("images",random.choice(image_list))
        files = {'file': open(img_path, 'rb')}
        self.client.post(
            "/",
            files=files,
        )

class LoadTest(HttpUser):
    tasks = [Face_Analysis_Task]
    wait_time = between(0,2)
    host = "http://localhost:8080"
    stop_timeout = 10
