import cv2
import numpy as np
import os
import uuid


def region_of_interest(img, vertices):

    """
    Applies a mask to the input image based on the specified region of interest (ROI).

    Args:
        img (numpy.ndarray): The input image.
        vertices (list of numpy.ndarray): Vertices defining the ROI polygon.

    Returns:
        numpy.ndarray: The masked image.
    """
    
    #takes an images and adds a mask to it
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 0, 255), thickness=3):
    """
    Draws lines on the input image.

    Args:
        img (numpy.ndarray): The input image.
        lines (list of numpy.ndarray): Lines defined by start and end points.
        color (tuple, optional): The color of the lines. Defaults to (0, 0, 255).
        thickness (int, optional): The thickness of the lines. Defaults to 3.
    """
        
    #taking an image and drawing a lines on it
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def enhance_white_color(image):

    """
    Enhances white regions in the input image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        tuple: A tuple containing the enhanced image, image with a dot at the center, 
        binary thresholded image, and the x and y coordinates of the center.
    """
        
    image_with_dot = image.copy()

    # Define lower and upper bounds for white color in HSV
    lower_white = np.array([237, 240, 237])  # Adjust these values as needed
    upper_white = np.array([255, 255, 255])  # Adjust these values as needed

    # Create a mask for white pixels within the specified range
    mask = cv2.inRange(image, lower_white, upper_white)

    # Apply the mask to extract white regions from the image
    white_regions = cv2.bitwise_and(image, image, mask=mask)

    # Convert the enhanced white regions back to the original color space
    enhanced_image = cv2.cvtColor(white_regions, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,5,3)

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)


    # Filter contours with area close to zero
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0.1 and cv2.minEnclosingCircle(cnt)[0][1] <= 170] 

    # Sort contours by area
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea)
    # Initialize center variable as None
    center = None
    if len(filtered_contours) > 0:
        smallest_contour = filtered_contours[0]
        # Calculate the center of the contour
        M = cv2.moments(smallest_contour)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        center = (center_x, center_y)
        # Print the center coordinates
        print("Center of the smallest contour (x, y):", center_x, center_y)
        cv2.circle(image_with_dot, center, 5, (0, 0, 255), -1)
    else:
        # Handle the case when no contour satisfies the filtering conditions
        center_x, center_y = None, None
        print("No contours found that satisfy the conditions.")
    
    return image, image_with_dot, thresh, center_x, center_y

def save_image_with_coordinates(image, image_with_dot, folder_path, folder_path2, x, y):

    """
    Saves the input image and the image with a dot at the center to the specified folder.

    Args:
        image (numpy.ndarray): The input image.
        image_with_dot (numpy.ndarray): The image with a dot at the center.
        folder_path (str): The path to the folder where the images will be saved.
        folder_path2 (str): The path to the second folder where the images will be saved.
        x (int): The x-coordinate of the center.
        y (int): The y-coordinate of the center.
    """

    # Generate a unique identifier (UUID)
    unique_id = str(uuid.uuid4())

    # Format the filename
    filename = f"xy_{x}_{y}_{unique_id}.jpg"

    # Create the full path to the image file
    file_path = os.path.join(folder_path, filename)
    file_path1 = os.path.join(folder_path2, filename)


    # Save the image with the specified path and filename
    cv2.imwrite(file_path, image)
    cv2.imwrite(file_path1, image_with_dot)



def process_images_in_folder(folder_path, new_folder_path, new_folder_path2):
    """
    Processes all the images in the specified folder.

    Args:
        folder_path (str): The path to the folder containing the images.
        new_folder_path (str): The path to the folder where the processed images will be saved.
        new_folder_path2 (str): The path to the second folder where the processed images will be saved.
    """

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for file_name in image_files:
        # Read the image
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)

        # Find lane lines in the image
        white, dot, thres, x, y = enhance_white_color(image)
        
        #save image with x and y coordinate
        save_image_with_coordinates(image, dot, new_folder_path, new_folder_path2, x, y)

        # Display the result
        cv2.imshow('Lane Lines', white)
        cv2.imshow('edges', thres)
        cv2.imshow('dot', dot)

        key = cv2.waitKey(0)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Your CLI description here')
    
    # Add three positional arguments (all of them are strings)
    parser.add_argument('image_dataset_captured', type=str, help='folder path for dataset to be processed', default= rf'.\images_path\image_dataset_captured')
    parser.add_argument('images_xy', type=str, help='folder path where the processed data is to be stored', default=rf'.\images_path\images_xy')
    parser.add_argument('images_with_circle', type=str, help='folder path for the images with the x and y marked in each image to cross reference if the points are correct',
                        default=rf'.\images_path\images_with_circle')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Access the three string arguments
    print("image_dataset_captured shape:", args.image_dataset_captured.shape)
    print("images_xy:", args.images_xy.shape)
    print("images_with_circle:", args.images_with_circle.shape)
    
    # # Provide the path to the folder containing the images
    # image_dataset_captured = r'C:\Users\AI_Admin\Downloads\jetbot\Jebot\image\photos'
    # images_xy = r'C:\Users\AI_Admin\Downloads\jetbot\Jebot\image\Images_xy'
    # images_with_circle = r'C:\Users\AI_Admin\Downloads\jetbot\Jebot\image\Images_with_circle'

    # Your logic goes here
    process_images_in_folder(args.image_dataset_captured, args.images_xy, args.images_with_circle)


if __name__ == '__main__':
    main()





