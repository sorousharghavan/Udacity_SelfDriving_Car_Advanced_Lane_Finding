from helper import *
from objects import Line
from moviepy.editor import VideoFileClip

def process_image(img):

    sob = sobel(img, (70, 255))
    s = hls_select(img, thresh=(180,255))
    binary_warped = np.zeros_like(sob)
    binary_warped[(sob == 1) | (s == 1)] = 1
    
    undist, binary_warped, M, Minv = corners_unwarp(binary_warped, mtx, dist, src=np.float32([[575,463], [705, 463], [1030, 666], [275, 666]]), dst=np.float32([[200,0], [1000,0], [1000,768], [200, 768]]))

    #plt.imshow(binary_warped)
    #plt.show()

    left_fit, right_fit, leftx, lefty, rightx, righty, ploty = find_lanes(binary_warped, left_lane.current_fit, right_lane.current_fit)

    left_lane.update(left_fit)
    right_lane.update(right_fit)
    
    find_curvature(leftx, lefty, rightx, righty, ploty)

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
#warped, m = corners_unwarp(mpimg.imread('./camera_cal/calibration10.jpg'), mtx, dist) 
#plt.imshow(warped)
#plt.show()

mask_region = np.array([[[575,463], [705, 463], [1030, 666], [275, 666]]])
#plt.imshow(region_of_interest(color_binary, mask_region))
#imgs = glob.glob('./test_images/test*.jpg')
#img = region_of_interest(mpimg.imread('./test_images/straight_lines1.jpg'), mask_region)

white_output = 'test.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


#for fname in imgs:
    #img = mpimg.imread(fname)
    #plt.imshow(img)
    #plt.show()


