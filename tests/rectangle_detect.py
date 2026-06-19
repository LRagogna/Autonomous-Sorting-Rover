"""Live Raspberry Pi camera test for detecting green objects.

HOW TO USE THIS FILE

1. Run this on the Raspberry Pi, not on a normal laptop.
2. Make sure the Raspberry Pi camera is connected and enabled.
3. Install the needed packages:

       sudo apt install python3-picamera2
       pip install -r requirements.txt

4. Start the script:

       python tests/rectangle_detect.py

5. Two windows should appear:

   - "Green Detection" shows the camera view with boxes around green objects.
   - "Mask" shows the black-and-white image the code uses to find green areas.

6. Press q to quit.

WHAT THIS FILE DOES

- The camera takes a picture over and over.
- Each picture is converted into a color format that is easier for OpenCV to
  search.
- The script keeps only green parts of the image.
- It finds large green blobs.
- It draws boxes around those blobs.

This is a proof-of-concept test, not the final rover vision system.
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import time


# Create the camera object. This is how Python talks to the Raspberry Pi camera.
picam2 = Picamera2()

# Configure the camera preview.
#
# RGB888 means every pixel stores red, green, and blue color values.
# 640x480 is small enough to process quickly, but still large enough to see
# objects clearly during testing.
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
)

# Start the camera and give it a moment to wake up. Cameras often need a short
# warm-up time before frames look stable.
picam2.start()
time.sleep(2)

print("Green detection running. Press q to quit.")

# This loop runs forever until the user presses q.
while True:
    # Grab one frame from the camera. A frame is one still image from the video
    # stream.
    frame = picam2.capture_array()

    # Picamera2 gives us RGB images. OpenCV usually expects BGR images, so we
    # convert RGB -> BGR first.
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # HSV stands for Hue, Saturation, Value.
    #
    # Hue is the "color family" part, like green or blue.
    # HSV makes color filtering easier than raw RGB/BGR because the color is
    # more separated from brightness.
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # These two arrays define what "green" means to the detector.
    #
    # If lighting changes or the object is a different shade of green, these
    # numbers may need tuning.
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # The mask is a black-and-white image:
    #
    # - white pixels mean "this part looks green"
    # - black pixels mean "this part does not look green"
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up noise in the mask.
    #
    # Erode removes tiny white specks.
    # Dilate grows the remaining white areas back out.
    # Together, this helps the detector ignore small bits of random noise.
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find separate white blobs in the mask. Each blob might be a green object.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Area is the size of this detected blob in pixels.
        area = cv2.contourArea(contour)

        # Ignore tiny blobs. This prevents little green reflections or noise
        # from being labeled as objects.
        if area < 500:
            continue

        # Turn the blob into a simple rectangle that surrounds it.
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the rectangle and label on the normal camera image so a human can
        # see what the code found.
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            frame_bgr,
            "Green Object",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Show the human-friendly camera view and the detector's black-and-white
    # mask at the same time.
    cv2.imshow("Green Detection", frame_bgr)
    cv2.imshow("Mask", mask)

    # waitKey(1) lets OpenCV update the windows and also checks keyboard input.
    # ord("q") is the number for the q key.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Stop the camera and close windows cleanly so the next run can use them again.
picam2.stop()
cv2.destroyAllWindows()
