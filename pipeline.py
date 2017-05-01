from helper import *
from objects import Line
from moviepy.editor import VideoFileClip

def process_image(img):
    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #sobel filter
    sob = sobel(gray, (70, 255))

    #saturation filter
    s = hls_select(img, thresh=(170,255))

    #grayscale threshold
    gray_bin = np.zeros_like(sob)
    gray_bin[gray >= 200] = 1

    #red filter
    r_filter = img[:,:,0]
    r_bin = np.zeros_like(sob)
    r_bin[r_filter >= 180]

    #combine all color and gradient filters
    binary_warped = np.zeros_like(sob)
    binary_warped[(sob == 1) | (s == 1) | (gray_bin == 1) | (r_bin == 1)] = 1

    #display for debugging
    #plt.imshow(binary_warped)
    #plt.show()
    
    #warp image into birds-eye view
    undist, binary_warped, M, Minv = corners_unwarp(binary_warped, mtx, dist, src=np.float32([[575,463], [705, 463], [1030, 666], [275, 666]]), dst=np.float32([[200,0], [1000,0], [1000,768], [200, 768]]))

    #find lines
    left_fit, right_fit, leftx, lefty, rightx, righty, ploty = find_lanes(binary_warped, left_lane.current_fit, right_lane.current_fit)

    #find the curvature of lines
    cleft, cright = find_curvature(leftx, lefty, rightx, righty, ploty)
    
    #off center distance
    off_centr = find_off_center(img.shape[1]/2, leftx[-1], rightx[-1])
    print("vehicle is off center by: ", off_centr)

    #sanity check
    if (cleft > 500 and cright > 500 and cleft * cright > 0):
        left_lane.update(left_fit, cleft)
        right_lane.update(right_fit, cright)
        
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    left_fit = left_lane.best_fit
    right_fit = right_lane.best_fit
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
	
    cv2.putText(result, 'left line curvature: ' + str(left_lane.radius_of_curvature) + ' m', (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 1)
    cv2.putText(result, 'right line curvature: ' + str(right_lane.radius_of_curvature) + ' m', (100,150), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 1)
    cv2.putText(result, "vehicle is off center by: " + str(off_centr) + ' m', (100,200), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 1)
    #return newwarp
    return result

#chessboard size
nx = 9
ny = 6

#lane objects
left_lane = Line()
right_lane = Line()

#calibration imageset path
cal_path = './camera_cal/calibration*.jpg'

ret, mtx, dist, rvecs, tvecs = calibrate_camera(cal_path, nx, ny)

white_output = 'test.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
