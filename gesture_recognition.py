import cv2
import numpy as np

def recognize_hand_gesture(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to binarize the image
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return "None"
    if len(contours) == 0:
        return "None"

    # Find the largest contour by area
    max_contour = max(contours, key=cv2.contourArea)

    # Check if the largest contour is big enough to be considered a hand (adjust this threshold as needed)
    if cv2.contourArea(max_contour) > 10000:
        # Convex hull
        hull = cv2.convexHull(max_contour)

        # Draw contours and hull
        cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)

        # Convexity defects
        defects = cv2.convexityDefects(max_contour, cv2.convexHull(max_contour, returnPoints=False))

        # Count defects (fingers)
        count_defects = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                # Draw circles at the points of convexity defects (optional)
                cv2.circle(frame, far, 5, (0, 0, 255), -1)

                # Calculate the angles between the points using the cosine rule
                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(end) - np.array(far))
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi

                # If angle is less than 90 degrees, count it as a defect (finger)
                if angle <= 90:
                    count_defects += 1

        # Based on the number of defects (fingers), classify the gesture
        if count_defects == 0:
            return "Hello"
        elif count_defects == 1:
            return "Yes"
        elif count_defects == 2:
            return "No"
        elif count_defects == 3:
            return "Thank You"
        elif count_defects == 4:
            return "Stop"

    return "None"
