import cv2
import numpy as np


class OpenCVCapture:
  """
  OpenCV wrapper to read from webcam.
  """
  def __init__(self):
    """
    Init.
    """
    self.cap = cv2.VideoCapture(0)

  def read(self):
    """
    Read one frame. Note this function might be blocked by the sensor.

    Returns
    -------
    np.ndarray
      Read frame. Might be `None` is the webcam fails to get on frame.
    """
    flag, frame = self.cap.read()
    if not flag:
      return None
    return np.flip(frame, -1).copy() # BGR to RGB
