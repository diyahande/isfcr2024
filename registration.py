
#### ORIGINAL CODE IS THE FIRST PART

# import cv2
# import os

# def preprocess_image(frame):
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Apply CLAHE for contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl1 = clahe.apply(gray)

#     # Apply Gaussian Blur
#     blurred = cv2.GaussianBlur(cl1, (5, 5), 0)

#     # Apply adaptive thresholding
#     thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY_INV, 11, 2)
#     return thresh 

# def main():
#     # Create directory to save images
#     save_dir = 'registered_user_images'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     # Open a connection to the webcam
#     cap = cv2.VideoCapture(0)
    
#     img_count = 0
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Preprocess the image
#         processed_frame = preprocess_image(frame)
        
#         # Display the resulting frame
#         cv2.imshow('Processed Webcam Feed', processed_frame)
        
#         # Save the processed frame
#         img_filename = os.path.join(save_dir, f'image_{img_count:04d}.jpeg')
#         cv2.imwrite(img_filename, processed_frame)
#         img_count += 1
        
#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # When everything done, release the capture
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



#-------------------------------------------------------2nd part one time registration----------------------------------------------------#

import cv2
import os

def save_image_file(img, output_path):
    cv2.imwrite(output_path, img)

def preprocess_image(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(cl1, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh
def capture_and_process_image(output_image_path):
    cap = cv2.VideoCapture(0)  # Open the webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        preprocessed_frame = preprocess_image(frame)

        # Display the preprocessed frame
        cv2.imshow('Preprocessed Image', preprocessed_frame)

        # Press 'c' to capture the image and save it
        if cv2.waitKey(1) & 0xFF == ord('c'):
            save_image_file(preprocessed_frame, output_image_path)
            print(f"Captured and saved image: {output_image_path}")
            break

        # Press 'q' to quit without saving
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()

output_image_path = "testing4.jpeg"  # Specify your output image path

capture_and_process_image(output_image_path)


# import numpy as np

# def preprocess_image(frame):
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Apply CLAHE for contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl1 = clahe.apply(gray)

#     # Apply Gaussian Blur
#     blurred = cv2.GaussianBlur(cl1, (5, 5), 0)

#     # Apply adaptive thresholding
#     thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY_INV, 11, 2)

#     # Apply morphological transformations for better edge detection
#     kernel = np.ones((2, 3), np.uint8)
#     morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    

#     return morph

# Example usage
# frame = cv2.imread('path_to_palmprint_image.jpg')
# processed_frame = preprocess_image(frame)
# cv2.imshow('Processed Palmprint', processed_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# import cv2
# import os

# def save_image_file(img, output_path):
#     cv2.imwrite(output_path, img)


#     ret,thresh1 = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
#     ret,thresh2 = cv2.threshold(frame,127,255,cv2.THRESH_BINARY_INV)
#     ret,thresh3 = cv2.threshold(frame,127,255,cv2.THRESH_TRUNC)
#     ret,thresh4 = cv2.threshold(frame,127,255,cv2.THRESH_TOZERO)
#     ret,thresh5 = cv2.threshold(frame,127,255,cv2.THRESH_TOZERO_INV)
#     cv2.imshow('Binary',thresh1)
#     cv2.imshow('Inverse Binary',thresh2)
#     cv2.imshow('Trunc',thresh3)
#     cv2.imshow('Tozero',thresh4)
#     cv2.imshow('Inverse Tozero',thresh5)

#  # Apply global thresholding
#     _, global_thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
#     cv2.imshow('Global Thresholding', global_thresh)

#     # Apply Otsu's thresholding
#     _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     cv2.imshow("Otsu's Thresholding", otsu_thresh)


#      # Apply Canny edge detection
#     edges = cv2.Canny(otsu_thresh, 50, 150, apertureSize=3)
#     cv2.imshow('Canny Edges', edges)

#     # Hough Line detection
#     lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
#     if lines is not None:
#         for line in lines:
#             rho, theta = line[0]
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             x1 = int(x0 + 1000 * (-b))
#             y1 = int(y0 + 1000 * (a))
#             x2 = int(x0 - 1000 * (-b))
#             y2 = int(y0 - 1000 * (a))
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

#     cv2.imshow('Hough Lines', frame)

#      # Probabilistic Hough Line detection
#     linesP = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
#     if linesP is not None:
#         for line in linesP:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
#     cv2.imshow('Hough Lines P', frame)


# def watershed_segmentation(img_path):
#     # img = cv2.imread(img_path)
#     # assert img is not None, "File could not be read, check with os.path.exists()"
#     gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('gray', gray)
    
#     # Apply Otsu's thresholding
#     ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     cv2.imshow('thresh', thresh)
#     # Noise removal
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#     cv2.imshow('morphologyEx', opening)
#     # Sure background area
#     sure_bg = cv2.dilate(opening, kernel, iterations=3)
#     cv2.imshow('dilate', sure_bg)
#     # Finding sure foreground area
#     dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

#     cv2.imshow('distance', dist_transform)

#     ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

#     cv2.imshow('sure_fg', sure_fg)

#     # Finding unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg, sure_fg)

#     cv2.imshow('subtract', unknown)

    
#     # Marker labelling
#     ret, markers = cv2.connectedComponents(sure_fg)

#     cv2.imshow('connected components', markers)

    
#     # Add one to all labels so that sure background is not 0, but 1
#     markers = markers + 1
    
#     # Mark the region of unknown with zero
#     markers[unknown == 255] = 0
#     markers = cv2.watershed(img_path, markers)
#     img_path[markers == -1] = [255, 0, 0]

#     cv2.imshow('Watershed Segmentation', img_path)





##################################   PRE PROCESS AN ALREADY EXISTING IMAGE    ############################



# import cv2
# import numpy as np

# def preprocess_image(frame):
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('Grayscale Image', gray)
#       # Wait for a key press to proceed

#     # Apply CLAHE for contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl1 = clahe.apply(gray)
#     cv2.imshow('CLAHE Image', cl1)

#     # Apply Gaussian Blur
#     blurred = cv2.GaussianBlur(cl1, (5, 5),4)
    
#     #testing bilateral filter
#     # blurred=cv2.bilateralFilter(cl1,9,75,75)
#     cv2.imshow('Blurred Image', blurred)
   
#     # Apply adaptive thresholding
#     thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                    cv2.THRESH_BINARY_INV, 11, 2)
#     cv2.imshow('Thresholded Image', thresh)
#     return thresh 

# def main():
#     # Load an image from file
#     img_path = '/Users/mallik/Desktop/OPENCV/captured_image_new.jpeg'  # Replace with your image file path
#     frame = cv2.imread(img_path)
    
#     if frame is None:
#         print("Error: Could not load image.")
#         return
    
#     # Preprocess the image and display each stage
#     preprocess_image(frame)
#     # watershed_segmentation(frame)
    
#     # Wait for a key press to close all windows
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()