import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 0, 255), thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def enhance_white_color(image):

    # Define lower and upper bounds for white color in HSV
    lower_white = np.array([237,240,237])  # Adjust these values as needed
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
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0.1]

    # Sort contours by area
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea)
    # Initialize center variable as None
    center = None
    smallest_contour = filtered_contours[0]

    # Calculate the center of the contour
    if smallest_contour is not None:
        M = cv2.moments(smallest_contour)
        # if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        center = (center_x, center_y)
        # Print the center coordinates
        print("Center of the smallest contour (x, y):", center_x, center_y)
        cv2.circle(image, center, 5, (0, 0, 255), -1)
    
    return image, thresh



import os 
# Provide the path to the folder containing the images
folder_path = r'C:\Users\AI_Admin\Downloads\jetbot\Jebot\image\photos'
# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


for file_name in image_files:
        # Read the image
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)

        # Find lane lines in the image
        white, thres = enhance_white_color(image)

        # Display the result
        cv2.imshow('Lane Lines', white)
        cv2.imshow('edges', thres)
        
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
cv2.destroyAllWindows()
