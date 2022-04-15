import cv2
import numpy as np

old_lines = 0
old_left_slope = -1.1   #advanced course
old_right_slope = 1.1   #advanced course
# old_left_intercept = 800
# old_right_intercept = -600


def canny_edge(image):
    coppied_img = np.copy(image)
    # ret, thresh = cv2.threshold(coppied_img, 150, 300, cv2.THRESH_BINARY)
    converted = cv2.cvtColor(coppied_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(converted, (5, 5), 0)
    canny_image = cv2.Canny(blur, 50, 150)
    return canny_image

def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array([
        # [(50, height),(1080, height),(600, 250)]
        [(30, height), (300, height), (550, 350), (650, 350), (900, height), (1080, height), (700, 350), (480, 350)]    #advanced course
    ])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, polygon, (255,255,255))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def make_coordinate(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int((image.shape[0]*9)/10)
    y2 = int(y1*(6/10))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    global old_left_slope, old_right_slope  #advanced course
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            if abs(slope - old_left_slope) < 0.2: #0.3 best for long    #advanced course
                left_fit.append((slope, intercept))
                old_left_slope = slope
        else:
            if abs(slope - old_right_slope) < 0.2: # 0.3 best for long  #advanced course
                right_fit.append((slope, intercept))
                old_right_slope = slope

    if len(left_fit) and len(right_fit):
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coordinate(image, left_fit_average)
        right_line = make_coordinate(image, right_fit_average)
        averaged_lines = [left_line, right_line]
        return averaged_lines

def display_lines(image, lines):
    if lines is not None:
        # heading point 계산을 위한 중간지점 x,y 좌표 구하기  #advanced course
        midpoint_x1 = int((lines[0][0] + lines[1][0]) * 0.5)
        midpoint_y1 = int((lines[0][1] + lines[1][1]) * 0.5)
        midpoint_x2 = int((lines[0][2] + lines[1][2]) * 0.5)
        midpoint_y2 = int((lines[0][3] + lines[1][3]) * 0.5)

        # 검출된 좌, 우측 차선 그리기
        for x1, y1, x2, y2 in lines:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 10)

        # heading선을 찾기 위한 가로선 그리기   #advanced course
        cv2.line(image, (lines[0][0], y1), (lines[1][0], y1), (0, 255, 0), 2)  # lower 선
        cv2.line(image, (lines[0][2], y2), (lines[1][2], y2), (0, 255, 0), 2)  # upper 선

        # heading선 그리기  #advanced course
        cv2.line(image, (midpoint_x1, midpoint_y1), (midpoint_x2, midpoint_y2), (0, 0, 255), 2)

        # target angle을 구하기 위한 기준점 표시
        # cv2.circle(image, (int(image.shape[1]*0.48), int((image.shape[0]*9)/10)), 3, (255, 0, 255), -1)
        # cv2.circle(image, (int(image.shape[1]*0.48), int(y1*(6/10))), 3, (255, 0, 255), -1)
        # target angle 구하기   #advanced course
        target_ang = round(np.rad2deg(-1*np.arctan((int(midpoint_x1 - midpoint_x2)) / int(midpoint_y1 - midpoint_y2))), 1)
        cv2.putText(image, 'target angle: {} deg'.format(target_ang), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)
    return image

# img = cv2.imread('motorway_image.jpg')
# canny = canny_edge(img)
# cropped_image = region_of_interest(canny)
# lines = cv2.HoughLinesP(cropped_image, 3, 1*np.pi/180, 130, np.array([]), maxLineGap=400)
# averaged_lines = average_slope_intercept(img, lines)
# line_image = display_lines(img, averaged_lines)
# cv2.imshow('lane detection', line_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("./video/motorway_techno.mp4")
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        canny = canny_edge(frame)
        cropped_image = region_of_interest(canny)
        lines = cv2.HoughLinesP(cropped_image, 3, 1*np.pi/180, 130, np.array([]), minLineLength=100, maxLineGap=400)
        try:
            averaged_lines = average_slope_intercept(frame, lines)
            line_image = display_lines(frame, averaged_lines)
            old_lines = averaged_lines
        except:
            line_image = display_lines(frame, old_lines)

        cv2.imshow('lane detection', line_image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("program has shutdown in purpose")
            break
    else:
        print("no more frame!")
        break

cap.release()
cv2.destroyAllWindows()