"""
    Module: image.py
    This module contains a declaration and definition of useful functions for taking images using my camera module.
"""
from numpy import float32
from camera import Camera


def unwrap(image_input, precision):
    """
        Function to unwrap extra array and convert to desired precision.

        Args:
            image_input (numpy.array): The numpy array containing the images taken via the camera.
            precision (numpy.dtype): The numpy dtype for the image precision.

        Returns:
            image_input (numpy.array): A numpy array containing the processed image.
    """
    # Taking image out of extra array wrapping.
    image_input = image_input[0]
    # Return image after converting to some precision.
    return image_input.astype(precision)


def take_images(camera_width, camera_height, precision=float32, num_images=1):
    """
        Function to take images using the first available camera and return them in a 1-dimensional array
    
        Args:
            camera_width (int): The camera width in pixels to use.
            camera_height (int): The camera height in pixels to use.
            precision (numpy.dtype): The numpy dtype for the image precision.
            num_images (int): The number of images to take.
        
        Returns:
            image (numpy.array): The numpy array containing the taken images.
    """
    # Taking the examples and storing them in "image".
    camera = Camera(camera_width, camera_height)
    image = camera.get_images(num_images)
    del camera
    image = unwrap(image, precision)
    return image
