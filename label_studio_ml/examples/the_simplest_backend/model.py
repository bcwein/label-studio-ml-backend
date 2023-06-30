import os
import numpy as np
import label_studio_sdk
from ultralytics import YOLO

from label_studio_ml.model import LabelStudioMLBase

LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8000')
LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', '787649668920d667e32e8a7a94a591062be1e3ae')
prefix = "/app/media/"

LABELS = {
    0: 'Henrica',
    1: 'asterias rubens',
    2: 'asteroidea',
    3: 'background',
    4: 'fucus vesiculosus',
    5: 'mytilus edulis',
    6: 'myxine glurinosa',
    6: 'pipe',
    7: 'rock',
    8: 'saccharina latissima',
    9: 'seafloor',
    10: 'tree',
    11: 'ulva intestinalis',
    12: 'urospora',
    13: 'zostera marina'
}

class MyModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.model = YOLO('/app/Sognefjord-Yolov8-2.pt')

    def predict(self, tasks, **kwargs):
        output_prediction = []
        for task in tasks:
            score = 0
            image_path =  task['data']['image']
            image_path = image_path.split("/", 2)[-1]
            image_path = prefix + image_path
            results = self.model.predict(image_path)
            for result in results:
                boxes = result.boxes
                masks = result.masks
                output_prediction_task = []
                for label, confidence, box, polygon in zip(boxes.cls, boxes.conf, boxes.xyxyn, masks.xyn):
                    label = int(label.item())
                    output_prediction_task.append({
                        'from_name': 'polygon_label',
                        'to_name': 'image',
                        'type': 'polygonlabels',
                        'score': confidence.item(),
                        'value': {
                            'polygonlabels': [LABELS[label]],
                            'points': (polygon*100).tolist()
                        }
                    })

                    x1, y1, x2, y2 = box
                    output_prediction_task.append({
                        'from_name': 'rectangle_label',
                        'to_name': 'image', 
                        'type': 'rectanglelabels',
                        'score': confidence.item(),
                        'value': {
                            'rectanglelabels': [LABELS[label]],
                            'x': float(x1*100),
                            'y': float(y1*100),
                            'width': float((x2 - x1)*100),
                            'height': float((y2 - y1)*100)
                        }
                    })

                    score += confidence
                    
            # Handle case where there are no predictions
            if len(results) > 0:
                score = float(score / len(boxes))
            else:
                score = 0.0

            output_prediction.append({
                'result': output_prediction_task,
                'score': score
            })

        return output_prediction
    

    def download_tasks(self, project):
        """
        Download all labeled tasks from project using the Label Studio SDK.
        Read more about SDK here https://labelstud.io/sdk/
        :param project: project ID
        :return:
        """
        ls = label_studio_sdk.Client(LABEL_STUDIO_HOST, LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project)
        tasks = project.get_unlabeled_tasks()
        return tasks