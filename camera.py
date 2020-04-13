"""
    Module: camera.py
    This module contains a declaration and definition of the Camera class for taking images.
"""

import cv2
import numpy as np


class Camera:
    def __init__(self, width, height, write_to_file=False, color=False):
        """
            The constructor function for a Camera object

            Args:
                width (int): The camera width in pixels to use.
                height (int): The camera height in pixels to use.
                write_to_file (boolean): Boolean value on whether to write the captured images to file.
                color (false): Boolean value on whether to capture color or use grayscale.
        """
        self.camera = cv2.VideoCapture(0)
        self.write_to_file = write_to_file
        self.color = color
        self.width = width
        self.height = height

        self.set_camera_parameter(3, width)
        self.set_camera_parameter(4, height)

        self.count_img = 0

        # Number of colors for RGB.
        self.rgb_const = 3

    def take_image(self):
        """
            Getting an image from the camera and processing it according to the parameters.

            Returns (numpy.array): A numpy array containing the captured image.
        """
        return_value, image = self.camera.read()

        if self.color and self.write_to_file:
            self.write_image(image)
        if not self.color:
            image = self.monochrome_image(image)
            if self.write_to_file:
                self.write_image(image)

        # Changing the image to a 1D array.
        flat = image.ravel()
        return flat

    def write_image(self, image):
        """
            A function to write the image to disk.

            Args:
                image (numpy.array): A numpy array containing the captured image.
        """
        cv2.imwrite("images/image%d.png" % self.count_img, image)

    @staticmethod
    def monochrome_image(image):
        """
            A function to make the image monochrome.
            
            Args:
                image (numpy.array): A numpy array containing the captured image.
            
            Returns:
                image (numpy.array): The same image stipped of all its color information.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_images(self, num_images):
        """
            Function to actually take images using the camera according to parameters.

            Args:
                num_images (int): The number of images to take.

            Returns:
                flat (numpy.array): The numpy array containing the images.
        """
        if self.color:  
            # Defining memory for monochrome image.
            flat = np.array(
                [np.zeros((self.width * self.height * self.rgb_const)) for i in range(num_images)])
        else:  
            # Defining memory for color image.
            flat = np.array([np.zeros((self.width * self.height))
                             for i in range(num_images)])

        # Taking the images.
        for i in range(num_images):
            flat[i] = self.take_image()
            self.count_img = self.count_img + 1

        return flat

    def set_camera_parameter(self, param_enum, value):
        """
            Setting OpenCV camera parameter values.

            Args:
                param_enum (int): The integer representing the parameter to change.
                value (data): The value to change the parameter to.
        """ 
        self.camera.set(param_enum, value)

"""
 Camera parameter enumerations:
   0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
   1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
   2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
   3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
   4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
   5. CV_CAP_PROP_FPS Frame rate.
   6. CV_CAP_PROP_FOURCC 4-character code of codec.
   7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
   8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve().
   9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
   10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
   11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
   12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
   13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
   14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
   15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
   16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
   17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported.
   18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras.
"""