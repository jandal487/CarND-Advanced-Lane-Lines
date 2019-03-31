###==================================================
# HELPER FUNCTIONS (Part 1)
###==================================================
import numpy as np
import glob # to read files from dir
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Visualization function to be used in the following stages
def plot_the_result(original_image, result_img, t1="Original Image", t2="Thresholded"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    f.tight_layout()
    ax1.imshow(original_image)
    ax1.set_title(t1, fontsize=30)
    ax2.imshow(result_img)
    ax2.set_title(t2, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	
# Calculate dst 4 points, considering offset, height & width of image
def get_dst(offset=20, w=1200, h=700):
    dst = np.float32([[offset, offset],
                      [w-offset, offset],
                      [w-offset, h-offset],
                      [offset, h-offset]])
    return dst

# Get undistorted image
def undistorted_img(original_image, mtx, dist):
    return cv2.undistort(original_image, mtx, dist, None, mtx)

# Function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if(orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return sbinary

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelxy = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    sbinary = np.zeros_like(scaled_sobelxy)
    sbinary[(scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sbinary

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # 5) Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output

# Combining thresholds: The code is taken from lessons
def get_comb_binarized(original_image):
	# Convert to HLS color space and separate the S channel
	# Note: img is the undistorted image
	hls = cv2.cvtColor(original_image, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	
	# Sobel x
	gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
	
	# Threshold x gradient
	thresh_min = 20
	thresh_max = 100
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
	
	# Threshold color channel
	s_thresh_min = 170
	s_thresh_max = 255
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

	# Combine the two binary thresholds
	combined_binary = np.zeros_like(sxbinary)
	combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
	
	return combined_binary
	
def get_comb_color_binarized(original_image):
    """
    Convert image to HSV color space and suppress any colors
    outside of the defined color ranges
    """
    hsv = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
    lower = np.array([20,60,60], dtype=np.uint8) 
    upper = np.array([38,174, 250], dtype=np.uint8)
    yellow_range = cv2.inRange(hsv, lower, upper)

    lower = np.array([202,202,202], dtype=np.uint8)
    upper = np.array([255,255,255], dtype=np.uint8) 
    white_range = cv2.inRange(original_image, lower, upper)
    # Threshold color_binary
    c_binary = np.zeros_like(yellow_range)
    c_binary[(yellow_range >= 1) | (white_range >= 1)] = 1
    
    # Saturation channel
    hls = cv2.cvtColor(original_image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Threshold s channel
    s_thresh = [90,255]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Sobel x
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sx_thresh = [20,100]
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(c_binary == 1) | (s_binary == 1) | (sx_binary == 1)] = 1
    
    return combined_binary

# Function to create pipeline of combination 
# of color and gradient thresholding
def get_bin_thresholding(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	
	# Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    return color_binary
	
# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

# Get Perspective transformation
def WarpPerspective(image, src, dst):
    y = image.shape[0]
    x = image.shape[1]
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (x, y), flags=cv2.INTER_LINEAR)

# Create histogram of image binary activations
def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram
	
def get_histogram_peaks(binary_warped, bottom_pct=0.5):
    # bottom_pct: How much of the bottom to use for initial tracer placement

    shape = binary_warped.shape

    bottom_sect = binary_warped[-int(bottom_pct*shape[0]):, :]

    left_peak = bottom_sect[:, :int(0.5*shape[1])].sum(axis=0).argmax()
    right_peak = bottom_sect[:, int(0.5*shape[1]):].sum(axis=0).argmax() + 0.5*shape[1]

    # Return x-position of the two peaks
    return left_peak, right_peak
	
###==================================================
# HELPER FUNCTIONS (Part 2)
###==================================================
# Implement Sliding Windows and Fit polynomial for left and right lanes
def get_sliding_windows(binary_warped,nwindows,margin,minpix,left_line,right_line):
    left_peak, right_peak = get_histogram_peaks(binary_warped)
    histogram = hist(binary_warped)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Sliding windows
    if (left_line.detected == False) or (right_line.detected == False) :
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        left_line.detected = True
        right_line.detected = True
    else:
        left_lane_inds = ((nonzerox > (left_line.current_fit[0] * (nonzeroy**2) + 
                                       left_line.current_fit[1] * nonzeroy + 
                                       left_line.current_fit[2] - margin)) & 
                          (nonzerox < (left_line.current_fit[0] * (nonzeroy**2) + 
                                       left_line.current_fit[1] * nonzeroy + 
                                       left_line.current_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_line.current_fit[0] * (nonzeroy**2) + 
                                        right_line.current_fit[1] * nonzeroy + 
                                        right_line.current_fit[2] - margin)) & 
                           (nonzerox < (right_line.current_fit[0] * (nonzeroy**2) + 
                                        right_line.current_fit[1] * nonzeroy + 
                                        right_line.current_fit[2] + margin)))
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    # Save successful fit of lines to prevent case with empty x, y        
    if (len(leftx) < 1500):
        leftx = left_line.allx
        lefty = left_line.ally
        left_line.detected = False
    else:
        left_line.allx = leftx
        left_line.ally = lefty
    if (len(rightx) < 1500):
        rightx = right_line.allx
        righty = right_line.ally
        right_line.detected = False
    else:
        right_line.allx = rightx
        right_line.ally = righty
    
    if(left_line.allx is None):
        left_line.allx = leftx
    if(left_line.ally is None):
        left_line.ally = lefty
    if(right_line.allx is None):
        right_line.allx = rightx
    if(right_line.ally is None):
        right_line.ally = righty
        
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
	
    # Sanity checks:
    if (left_line.current_fit[0] == False):
        left_line.current_fit = left_fit
        right_line.current_fit = right_fit
    
    if (abs(left_line.current_fit[1] - left_fit[1]) > 0.18):
        left_line.current_fit = left_line.best_fit
        left_line.detected = False
    else:
        left_line.current_fit = left_fit
        if left_line.recent_xfitted: #if empty_list will evaluate as false.
            left_line.recent_xfitted.pop()
        left_line.recent_xfitted.insert(0, left_line.current_fit)
        avg = np.array([0,0,0], dtype='float')
        for element in left_line.recent_xfitted:
            avg = avg + element
        left_line.best_fit = avg / (len(left_line.recent_xfitted))
        
    if (abs(right_line.current_fit[1] - right_fit[1]) > 0.18):
        right_line.current_fit = right_line.best_fit
        right_line.detected = False
    else:
        right_line.current_fit = right_fit
        if right_line.recent_xfitted: #if empty_list will evaluate as false.
            right_line.recent_xfitted.pop()
        right_line.recent_xfitted.insert(0, right_line.current_fit)
        avg = np.array([0,0,0], dtype='float')
        for element in right_line.recent_xfitted:
            avg = avg + element
        right_line.best_fit = avg / (len(right_line.recent_xfitted))
        
    if (abs(right_line.current_fit[1] - right_fit[1]) > 0.38 and
        abs(left_line.current_fit[1] - left_fit[1]) < 0.1):
        right_line.current_fit[0] = left_line.current_fit[0]
        right_line.current_fit[1] = left_line.current_fit[1]
        right_line.current_fit[2] = left_line.current_fit[2] + 600
        if right_line.recent_xfitted: #if empty_list will evaluate as false.
            right_line.recent_xfitted.pop()
        right_line.recent_xfitted.insert(0, right_line.current_fit)
        avg = np.array([0,0,0], dtype='float')
        for element in right_line.recent_xfitted:
            avg = avg + element
        right_line.best_fit = avg / (len(right_line.recent_xfitted))
        
    if (abs(left_line.current_fit[1] - left_fit[1]) > 0.38 and
        abs(right_line.current_fit[1] - right_fit[1]) < 0.1):
        left_line.current_fit = left_fit
        if left_line.recent_xfitted: #if empty_list will evaluate as false.
            left_line.recent_xfitted.pop()
        left_line.recent_xfitted.insert(0, left_line.current_fit)
        avg = np.array([0,0,0], dtype='float')
        for element in left_line.recent_xfitted:
            avg = avg + element
        left_line.best_fit = avg / (len(left_line.recent_xfitted))
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = (left_line.current_fit[0] * ploty**2 + 
                 left_line.current_fit[1] * ploty + 
                 left_line.current_fit[2])
    right_fitx = (right_line.current_fit[0] * ploty**2 + 
                  right_line.current_fit[1] * ploty + 
                  right_line.current_fit[2])
        
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    return out_img,left_fitx,right_fitx,ploty,left_fit,right_fit

def visualize_lanes_slidingWindows(slidingWindows, left_fitx, right_fitx, ploty):
    # Plot the polynomial lines onto the image
    plt.imshow(slidingWindows)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

# Alternative solution to sliding windows
def get_smooth_lanes(binary_warped, margin, left_fit, right_fit):

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
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
    #left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    img_shape = binary_warped.shape
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
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
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    smoothLanes = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return smoothLanes, left_fitx, right_fitx, ploty
	
def visualize_smooth_lanes(smoothLanes, left_fitx, right_fitx, ploty):
    # Plot the polynomial lines onto the image
    plt.imshow(smoothLanes)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
	
def get_realRadiusOfCurvature(binary_warped, left_fit, right_fit):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty +left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    carPosition = binary_warped.shape[1]/2
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    y_eval=np.max(ploty)
    
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    
    actualPosition = (left_lane_bottom + right_lane_bottom)/2
    
    distance = (carPosition - actualPosition)* xm_per_pix
    
    return (left_curverad + right_curverad)/2, distance
	
	
def unwarp_image(binary_warped, original_image, src, dst, left_fit, right_fit):
    # Calculate inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty +left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_warped.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 

    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, newwarp, 0.5, 0)
    return result