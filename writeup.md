## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `Stage0_Camera_Calibration.ipynb`, notebook.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objPoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgPoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objPoints` and `imgPoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

The results are also stored in this directory: `/camera_cal/`. The camera calibration and distortion coefficients are stored using pickle to be used later on in other notebooks.

### Pipeline (single images)

The code for this step is contained in 2 notebooks to show results clearly. The corresponding notebooks are `Stage1_Pipeline_a.ipynb.ipynb` & `Stage1_Pipeline_b.ipynb.ipynb`. The `Stage1_Pipeline_a.ipynb.ipynb` shows results of undistortion, thresholding image with several techniques and finally with perspective transform example. The `Stage1_Pipeline_b.ipynb.ipynb` is all about finding histogram from warped image, getting sliding windows, getting smooth lanes, finding radius of curvature and unwarping to original image. Both notebooks are well commented and pretty self explanatory. Here I will quickly explain the necessary points. Additionally, in order to work cleanly, I have written helper functions in a separate python file. I call this file as `pipeline_helper_functions.py`.

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images as shown in `Code cell # 3` of `Stage1_Pipeline_a.ipynb.ipynb` notebook. 

One example image is shown here:
![GitHub Logo](/output_images/test_img_undistorted_6.jpg)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Following the example in lessons, I created a function by the name of `get_comb_color_binarized()`. It can be found in `pipeline_helper_functions.py` at `line # 126` and can also be found in `Code cell # 8` of `Stage1_Pipeline_a.ipynb.ipynb`. The resulting image is also show in `Code cell # 8` of `Stage1_Pipeline_a.ipynb.ipynb` notebook.

One example image is shown here:
![GitHub Logo](/output_images/test_img_combined_img_6.jpg)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `WarpPerspective()`, which appears at `lines 208 through 212` in the file `pipeline_helper_functions.py` (or, for example, in the `2nd last code cell` of `Stage1_Pipeline_a.ipynb.ipynb` notebook).  The `WarpPerspective()` function takes as inputs an image (`combined_img`), as well as source (`src`) and destination (`dst`) points.  I tried several src and dst values and finally hard coded these values:

```python
src = np.float32([[550, 470], [760, 470], [1125, 670], [200, 670]])
dst = np.float32([[10, 10], [1200, 10], [1200, 700], [10, 700]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 550, 470      | 10, 10        | 
| 760, 470      | 1200, 10      |
| 1125, 670     | 1200, 700      |
| 200, 670      | 10, 700        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. The resulting image is also show in 'Code cell # 13' of `Stage1_Pipeline_a.ipynb.ipynb` notebook.

One example image is shown here:
![GitHub Logo](/output_images/test_img_wrapPerspective_6.jpg)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The line detection code could be found at `Code cell # 6, 7 & 8` of `Stage1_Pipeline_b.ipynb.ipynb` notebook. I implemented sliding windows algorithm and then implemented (alternative approach), smooth lanes detection algorithm. Sliding windows algorithm calculates the histogram on the X axis. First of all, I initialize left and right lanes with a `Lane()` class, as mentioned in the lessons. I find peaks from histogram and by sliding windows I collect pixels for left and right lane. Then I fit a second order polynomial for each of the lanes. The code can be found in function `get_sliding_windows()`, in `pipeline_helper_functions.py`, starting from `line # 243 to 414`. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

`Stage1_Pipeline_b.ipynb.ipynb` has implementation of this at `Code cell 8 & 9`. Here I have 2 functions, `measure_curvature_pixels()` & `get_realRadiusOfCurvature()` to get values in pixels & meters. The result of this is executed on 1 example image, that can be found at `Code cell 8 & 9`. In this notebook, I am using same image through out, therefore, left_fit and right_fit is used as hard coded for every step.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

`Stage1_Pipeline_b.ipynb.ipynb` has implementation of this at `Code cell 10& 11`. Here I have a function, `unwarp_image()` that get original_img, combined_img, left_fit, right_fit, src, dst values and gives unwarped image as output. The result of this is executed on 1 example image, that can be found at `Code cell 12` of `Stage1_Pipeline_b.ipynb.ipynb` notebook. I am using this `unwarp_image()` in `Stage1_Pipeline_c.ipynb.ipynb` to create the `pipeline()`, that is, executed for every frame of the video in next task. 

One example image, as result of `unwarp_image()` from `Stage1_Pipeline_c.ipynb.ipynb` is below:
![GitHub Logo](/output_images/test_img6.jpg)

---

### Pipeline (video)

The notebook that contains pipeline and code to generate output videos is: `Stage1_Pipeline_c.ipynb.ipynb`. This notebook is executed on all 3 videos, those are: `project_video.mp4`, `challenge_video.mp4`, & `harder_challenge_video.mp4`. The output of all these videos can be found in `/output_videos/` directory of my github respository. 

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/jandal487/CarND-Advanced-Lane-Lines/blob/master/output_videos/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

 - It is very difficult to solve challenge videos, especially, when there are sharp turns. Currently, result of challenge videos is not satisfactory. I will spend more time on them and come up with accurate solutions.
 - The project video is very accurate. However, I have still few things to improve in my code. I should apply smoothing between frame changes so that the result is smooth and stable. 
  - I should also spend more time on color spaces and find a better function to get detect yellow, white lanes in different contrast, light and dark images. 
 - Since this project is heuristic based, as we have to find right src points and hard code them in the algorithm, so overall the project has been very time consuming. However, it has been very interesting, especially, the last review was very helpful to improve my results. 
