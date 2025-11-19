from ultralytics import YOLO
import torch
import pandas as pd
import numpy as np
import cv2

names = ['Front top left', 'Front bottom left', 'Front top middle', 'Front bottom middle', 'Front top right', 'Front bottom right', 'Back divider top', 'Front divider top', 'Back top left', 'Back top right']

class BoxDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def detect(self, image):
        results = self.model.predict(image, device=self.device)
        if len(results[0].keypoints) == 0:
            return False, None
        xy = results[0].keypoints.xy[0].cpu().numpy()

        result = {names[i]:xy[i] for i in range(len(names))}

        result = pd.Series(result)

        if np.any(np.all(xy == 0, axis=1)):
            return False, result

        return True, result
    
    def draw_keypoints(self, image, detection_result: pd.Series):
        for name, point in detection_result.items():
            if np.all(point == 0):
                continue
            cv2.circle(image, tuple(point.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(image, name, tuple(point.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return image
    
    def _all_non_zero(self, detection_result: pd.Series):
        return np.all(np.array(detection_result.to_list()) != 0).all()
    
    def guess_missing_keypoints(self, detection_result: pd.Series, frame_width, frame_height):

        if detection_result['Front top right'].sum() == 0 and self._all_non_zero(detection_result.drop('Front top right')):

            x = detection_result['Front bottom right'][0]
            y = detection_result['Front bottom right'][1]
            y -= detection_result['Front bottom left'][1] - detection_result['Front top left'][1]
            y = max(y, 0)
            detection_result['Front top right'] = np.array([x, y])

            return detection_result
        
        elif detection_result['Front bottom right'].sum() == 0 and self._all_non_zero(detection_result.drop('Front bottom right')):

            x = detection_result['Front top right'][0]
            y = detection_result['Front top right'][1]
            y += detection_result['Front bottom left'][1] - detection_result['Front top left'][1]
            y = min(y, frame_height)
            detection_result['Front bottom right'] = np.array([x, y])

            return detection_result
        
        elif detection_result['Front top right'].sum() == 0 and detection_result['Front bottom right'].sum() == 0 and self._all_non_zero(detection_result.drop(['Front top right', 'Front bottom right'])):
            # predict the front bottom right
            width = abs(detection_result['Front bottom left'][0] - detection_result['Front bottom middle'][0])
            # simply assume x is same lenth from middle to right
            x = detection_result['Front bottom middle'][0] + width
            x = min(x, frame_width)

            m = (detection_result['Front bottom middle'][1] - detection_result['Front bottom left'][1]) / (detection_result['Front bottom middle'][0] - detection_result['Front bottom left'][0])
            c = detection_result['Front bottom left'][1] - m * detection_result['Front bottom left'][0]
            y = m * x + c
            y = min(y, frame_height)

            detection_result['Front bottom right'] = np.array([x, y])

            # predict the front top right
            y -= detection_result['Front bottom left'][1] - detection_result['Front top left'][1]

            # assume same x as bottom right
            detection_result['Front top right'] = np.array([x, y])

            return detection_result
        
        return None
    
    def get_pixel_to_cm_conversion_factor(self, detection_result: pd.Series):
        '''
        calculate by height of the divider
        does not adjust for perspective
        '''
        divider_height = abs(detection_result['Front divider top'][1] - detection_result['Front top middle'][1])
        factor = 17.018 / divider_height
        return factor

    def start_logging(self, savePath):
        print("BoxDetector logging started.")
        self.dataFrame = pd.DataFrame(columns=['Timestamp', 'FrameIdx'] + names)
        self.savePath = savePath

    def append(self, detection_result: pd.Series, time, frameidx):
        # create a copy to avoid modifying the original
        detection_result_copy = detection_result.copy()
        
        detection_result_copy = detection_result_copy.apply( lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        # add the timestamp and frame index to the pd.series copy
        detection_result_copy['Timestamp'] = time
        detection_result_copy['FrameIdx'] = frameidx
      
        self.dataFrame = pd.concat([self.dataFrame, pd.DataFrame([detection_result_copy])], ignore_index=True)

    def close_log(self):
        print(f"Saving the dataFrame to {self.savePath}")
        self.dataFrame.to_csv(self.savePath, index=False)
        print("BoxDetector logging closed.")


