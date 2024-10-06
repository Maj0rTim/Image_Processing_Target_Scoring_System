import cv2
import numpy as np

def display_image(image):
    scale = 0.3
    image = cv2.resize(image, None, fx=scale, fy=scale)
    cv2.imshow("Targets", image)
    cv2.waitKey(0)
    cv2.destroyWindow("Targets")

def find_targets(original_image):
    grey = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(grey, 31)

    circles = cv2.HoughCircles (
        image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=500,
        param1=150,
        param2=30,
        minRadius=500, 
        maxRadius=1500
    )

    if circles is not None:
        targets = []
        rings = []
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            black = np.full(image.shape, 0, dtype="uint8")
            cv2.circle(black, (i[0], i[1]), i[2], (255, 255, 255), -1)
            target = cv2.bitwise_and(grey, grey, mask=black)
            ring, blank = get_scoring_rings(image, i[0], i[1], i[2])
            rings.append(ring)
            targets.append((target, ring, blank)) 
        return targets
    else:
        print("No targets Found!")
        exit()

def get_scoring_rings(image, X, Y, R):
    blank = np.full(image.shape, 0, dtype="uint8")
    rings = []
    offset = int(R/7.5)
    distance = offset
    for i in range(6):
        rings.append((X, Y, distance))
        cv2.circle(blank, (X, Y), distance, (255, 255, 255), 2)
        distance += offset
    return rings, blank

def refine_target(image):
    #WIP
    return

def score_bullets(coords, rings): 
    num = 0
    score = 0
    iteration = 10
    for coord in coords:
        i, j = (coord[0][0][0], coord[0][0][1 ])
        for X, Y, R in rings:
            if ((i - X) * (i - X) + (j - Y) * (j - Y) <= R * R):
                num += 1
                score += iteration 
                iteration -=1
                break
            else: 
                continue
    return score, num
            
def already_counted(contour, coords): 
    contour_x, contour_y = contour[0][0][0], contour[0][0][1]
    for x, y in coords:
        X, Y = x[0][0], y[0][1] 
        if abs(contour_x - X) <= 15 and abs(contour_y - Y) <= 15:
            return True
    return False

 
def get_bullet_coords(image):
    image = cv2.medianBlur(image, 27)
    image = cv2.bitwise_not(image)
    mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    coords = []
    blank = np.full(image.shape, 0, dtype='uint8')
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 700 or area > 2600:
            continue
        if already_counted(contour, coords):
            continue
        
        cv2.drawContours(blank, contour, -1, (255, 255, 255), 5)
        coords.append((contour[0],contour[1]))
    return coords, blank

def main():
    original = cv2.imread("Targets/TargetPhotos/20140428_193219.jpg") 
    #original = cv2.imread("Targets/TargetPhotos/20140412_164751.jpg")
    #original = cv2.imread("Targets/TargetPhotos/20131005_093504.jpg")
    #original = cv2.imread("Targets/TargetPhotos/20140405_175354.jpg")
    #original = cv2.imread("Targets/TargetPhotos/20131019_103203.jpg")

    scale = 0.2
    scaled = cv2.resize(original, None, fx=scale, fy=scale)
    cv2.imshow("original", scaled)
    targets = find_targets(original)
    for image, rings, blank in targets:
        display_image(image)
        display_image(blank)
        coords, blank = get_bullet_coords(image)
        display_image(blank)
        score, num = score_bullets(coords, rings)
        print("====================== ")
        print(f"Target Score: {score}")
        print(f"Number of Bullets: {num}")
        print("======================")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main() 

    