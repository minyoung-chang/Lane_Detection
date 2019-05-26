# import necessary packages
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def color_thresh(img, thresh=(0, 255)):
    # Convert to HLS and extract s-channel
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = img_hls[:,:,2]

    # Create a binary image of ones where threshold is met, zeros otherwise
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

#    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    # Return the binary image
    return s_binary

def combined_thresh(img):
    dir_binary = dir_thresh(img, constant.dir_kernels, constant.dir_thresholds)
    mag_binary = mag_thresh(img, constant.mag_kernels, constant.mag_thresholds)
    color_binary = color_thresh(img, constant.color_thresholds)

    combined_binary = np.zeros_like(color_binary)
    combined_binary[((mag_binary==1) & (dir_binary==1)) | (color_binary==1)]=1
    return combined_binary

# input image should be undistorted beforehand
def warp(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    # get transform matrix, M
    M = cv2.getPerspectiveTransform(src, dst)
    # warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size)

    return warped

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    #leftx_base = np.argmax(histogram[:midpoint])
    leftx_base = np.argmax(histogram[150:350])
    leftx_base += 150
    #rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    rightx_base = np.argmax(histogram[900:1100])
    rightx_base += 900

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//constant.nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    binary_warped_h = binary_warped.shape[0]

    # Step through the windows one by one
    for num in range(constant.nwindows):
        # Identify window boundaries in x and y (and right and left)
        window_y_bot = binary_warped_h - window_height*num
        window_y_top = binary_warped_h - window_height*(num+1)

        window_leftx_left = leftx_current - constant.margin    # x coords for the left lane
        window_leftx_right = leftx_current + constant.margin
        window_rightx_left = rightx_current - constant.margin    # x coord for the right lane
        window_rightx_right = rightx_current + constant.margin

        # Draw the windows on the visualization image
#         cv2.rectangle(out_img,(window_leftx_left,window_y_bot),
#         (window_leftx_right,window_y_top),(0,255,0), 2)
#         cv2.rectangle(out_img,(window_rightx_left,window_y_bot),
#         (window_rightx_right,window_y_top),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the current window
        good_left_inds = ((nonzeroy >= window_y_top) & (nonzeroy < window_y_bot) &
        (nonzerox >= window_leftx_left) &  (nonzerox < window_leftx_right)).nonzero()[0]

        good_right_inds = ((nonzeroy >= window_y_top) & (nonzeroy < window_y_bot) &
        (nonzerox >= window_rightx_left) &  (nonzerox < window_rightx_right)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > constant.minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > constant.minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]


    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return left_fitx, right_fitx, ploty, left_fit, right_fit, out_img


def fit_poly_search(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fitx, right_fitx, left_fit, right_fit, ploty

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 50

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, left_fit, right_fit, ploty = fit_poly_search(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    return leftx, lefty, rightx, righty, left_fitx, right_fitx, left_fit, right_fit, ploty, result


def curvature(leftx, lefty, rightx, righty, ploty):
    # calculate radius of curvature
    y_eval = np.max(ploty)

    left_fit_cr = np.polyfit(lefty*constant.ym_per_pix, leftx*constant.xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*constant.ym_per_pix, rightx*constant.xm_per_pix, 2)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*constant.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*constant.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

def offcenter(left_fit, right_fit, width, y):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # calculate off center
    left_pos = left_fit[0]*(y**2) + left_fit[1]*y + left_fit[2]
    right_pos = right_fit[0]*(y**2) + right_fit[1]*y + right_fit[2]
    center_pos = (left_pos + right_pos) / 2
    #width = result.shape[1]
    center_cam = width/2
    off = np.absolute(center_cam - center_pos)
    off_real = off * xm_per_pix

    return off_real

def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_undist = cv2.undistort(img, camera.mtx, camera.dist, None, camera.mtx)
    img_undist = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)

    # Find region of interest
    src = np.float32([[250,660], [530,480], [790,480], [1120,660]])
    dst = np.float32([[200,700], [200,0], [1100,0], [1100,700]])

    warped = warp(img_undist, src, dst)
    binary_warped = combined_thresh(warped)

    if left_lane.detected == False:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
        left_fitx, right_fitx, ploty, left_fit, right_fit, out_img_polydrawn = fit_polynomial(binary_warped)
    else:
        leftx, lefty, rightx, righty, left_fitx, right_fitx, left_fit, right_fit, ploty, out_img = search_around_poly(binary_warped, left_lane.previous_fit, right_lane.previous_fit)

    left_lane.previous_fit = left_fit    # update for next frame
    right_lane.previous_fit = right_fit

    left_lane.detected = True
    right_lane.detected = True

    img_size = (out_img.shape[1],out_img.shape[0])
    M_inv = cv2.getPerspectiveTransform(dst, src)   # dst and src reversed
    warped = cv2.warpPerspective(out_img, M_inv, img_size)

    #background = img
    background = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = warped
    added_image = cv2.addWeighted(background,0.7,overlay,1,0)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(added_image, 1, newwarp, 0.3, 0)

    left_curverad, right_curverad = curvature(leftx, lefty, rightx, righty, ploty)

    width = img.shape[1]
    y = img.shape[0]
    off_real = offcenter(left_fit, right_fit, width, y)

    # Write some Text
    write_on_frame(result, 'Left Radius of Curv. =', left_curverad, (50,50))
    write_on_frame(result, 'Right Radius of Curv. =', right_curverad, (50,100))
    write_on_frame(result, 'Off Center =', off_real, (50,150))

    return result

def write_on_frame(img, str, value, location):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1.25
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(img, str+'%.2f m' %value,
        location,
        font,
        fontScale,
        fontColor,
        lineType)
    return

class Constants:
    def __init__(self):
        # combined_thresh
        self.dir_thresholds = (0.7,0.9)
        self.dir_kernels = 7
        self.mag_thresholds = (25, 255)
        self.mag_kernels = 25
        self.color_thresholds = (125, 255)

        # offcenter & curvature
        self.ym_per_pix = 30/720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # find_lane_pixels
        self.nwindows = 9   # Choose the number of sliding windows
        self.margin = 80    # width of the windows +/- margin
        self.minpix = 40    # minimum number of pixels found to recenter window

class Camera_calibration:
    def __init__(self, image_directory):
        self.cam_cal_images = glob.glob(image_directory)
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = self.calibrate(self.cam_cal_images)

    def calibrate(self, images):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)  # number of boundaries in rows and columns
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)  # -1: unknown, # 2: make three (0,1,2) columns

        objpoints = []
        imgpoints = []
        img = cv2.imread(images[0])
        img_size = (img.shape[1], img.shape[0])

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        return ret, mtx, dist, rvecs, tvecs

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []

        self.previous_fit = [np.array([False])]
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

left_lane = Line()
right_lane = Line()

# Set up camera calibration
camera = Camera_calibration('camera_cal/calibration*.jpg')

# Set up constants
constant = Constants()

# Read in
img = mpimg.imread('test_images/test3.jpg')

''' CHECKING A FRAME'''
# result = process_image(img)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# ax1.imshow(img)
# ax1.plot(250, 660,'o')  # bottom left
# ax1.plot(530, 480,'o')  # top left
# ax1.plot(790, 480,'o')  # top right
# ax1.plot(1120, 660,'o')  # bottom right
# ax1.set_title('original', fontsize=30)
# ax2.imshow(result)
# ax2.set_title('result', fontsize=30)
# plt.show(block=False)
# plt.pause(3)

'''CREATING OUTPUT VIDEO'''
#Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

output = 'project_video_output1.mp4'
# To do so add .subclip(start_second,end_second) to the end of the line below
# Where start_second and end_second are integer values representing the start and end of the subclip
# You may also uncomment the following line for a subclip of the first 5 seconds

clip1 = VideoFileClip('project_video.mp4')#.subclip(40,43)
clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
clip.write_videofile(output, audio=False)
