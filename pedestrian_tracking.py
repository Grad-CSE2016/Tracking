import cv2
import sys, getopt
import blobs
import numpy as np
import time

#  Adjust all aspects of the configuration for this module at pt_config
import pt_config

def show_video(argv):
	"""
	Main Function for all video processing.  Defaults for this file are adjusted here.
	"""
	tracker = blobs.BlobTracker()

	#  Default Options for Running in Demo Mode
	video = "demo.avi"
	background = "demo_0.png"
	output =  "blob"
	method = "mog"

	file_name_base = ""

	try:
		opts,args = getopt.getopt(argv, "v:b:o:m:")
	except getopt.GetoptError:
		print "Getopt Error"
		exit(2)

	for opt, arg in opts:
		if opt == "-v":
			video = arg
		elif opt == "-b":
			background = arg
		elif opt == "-o":
			output = arg
		elif opt == "-m":
			method = arg


	masks = pt_config.masks

	print video , " " , background , " " , output

	file_name_base = "results/" + video.split("/")[-1].split(".")[-2] + "_" + method

	c = cv2.VideoCapture(video)
	_,f = c.read()

	if method == "ext":
		#  Use a predetermined background image
		c_zero = cv2.imread(background)
		#c_zero = f
	else:
		#  Use the growing accumulated average
		c_zero = np.float32(f)

	c.set(0, 000.0)
	height, width, _ = f.shape
	fps = c.get(5)
	fourcc = c.get(6)
	frames = c.get(7)

	#  Print out some initial information about the video to be processed.
	print fourcc, fps, width, height, frames

	if method == "mog":
		#  Setup MOG element for generated background subtractions
		bgs_mog = cv2.createBackgroundSubtractorMOG2()

		# MOG Erosion.Dilation
		for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(pt_config.mog_er_w, pt_config.mog_er_h))
		for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(pt_config.mog_di_w, pt_config.mog_di_h))


	orange = np.dstack(((np.zeros((height, width)),np.ones((height, width))*128,np.ones((height,width))*255)))
	ones = np.ones((height, width, 3))

	trails = np.zeros((height, width, 3)).astype(np.uint8)
	start_t = time.clock()
	current_frame = 0

	while 1:

		#s = raw_input()

		#  Get the next frame of the video
		_,f = c.read()

		#  Do some caculations to determin and print out progress.
		current_frame = c.get(1)
		t = time.clock()-start_t
		remainder = 1.0 - current_frame/float(frames)
		current = current_frame/float(frames)
		remaining = int(((t/current)*remainder) / 60.0)

		if int(current_frame)%25==0:
			print "Percentage: " , int((current_frame/frames)*100) , " Traces: " , len(tracker.traces), "Time left (m): ", remaining



		grey_image = bgs_mog.apply(f)
		#  Turn this into a black and white image (white is movement)
		thresh, im_bw = cv2.threshold(grey_image, 225, 255, cv2.THRESH_BINARY)



		# TODO Add Booleans to show or hide processing images
		#this wasnot hiddien cv2.imshow("Threshholded Image", im_bw)
		#cv2.imshow("Background", im_zero)
		#cv2.imshow('Background Subtracted', d1)
		#cv2.imshow("Thresholded", im_bw)


		#  Erode and Dilate Image to make blobs clearer.  Adjust erosion and dilation values in pt_config
		im_er = cv2.erode(im_bw, for_er)
		im_dl = cv2.dilate(im_er, for_di)

		# mask out ellipitical regions
		for mask in masks:
			cv2.ellipse(im_dl, (mask[0], mask[1]), (mask[2], mask[3]), 0, 0, 360, (0,0,0), -1)

		cv2.imshow("Eroded/Dilated Image", im_dl)

		_, contours, hierarchy = cv2.findContours(im_dl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		my_blobs = []
		for cnt in contours:
			try:
				x,y,w,h = cv2.boundingRect(cnt)
				#this is the rectangulars
				cv2.rectangle(f, (x,y), (x+w, y+h), (255,0,0), 2)

				moments = cv2.moments(cnt)
				x = int(moments['m10'] / moments['m00'])
				y = int(moments['m01'] / moments['m00'])

				my_blobs.append((x,y))
			except:
				print "Bad Rect"

		if len(my_blobs) > 0:
			tracker.track_blobs(my_blobs, [0,0,width,height], current_frame)

			for v in tracker.virtual_blobs:
				size = 5
				if v.got_updated:
					size = 10
				#cv2.rectangle(f, (int(v.x),int(v.y)), (int(v.x+size), int(v.y+size)), v.color, size)


		if pt_config.draw_video:

			for id in tracker.traces:

				ox = None
				oy = None

				if len(tracker.traces[id])>2:
					for pos in tracker.traces[id][-3:]:

						x = int(pos[0])
						y = int(pos[1])

						if ox and oy:
							sx = int(0.8*ox + 0.2*x)
							sy = int(0.8*oy + 0.2*y)

							#  Colours are BGRA
							#cv2.line(trails, (sx,sy), (ox, oy), (0,128,255), 1)
							#cv2.line(trails, (sx, sy), (ox, oy), (0,0,255), 2)

							oy = sy
							ox = sx
						else:
							ox,oy = x,y

			cv2.add(f,trails,f)
			cv2.drawContours(f,contours,-1,(0,255,0),1)



			cv2.imshow('output',f)
			cv2.waitKey(delay=1)



		#  Kill switch
		k = 0
		k = cv2.waitKey(1)

		if k == 27: # escape to close
			print "We're QUITING!"
			break

	cv2.destroyAllWindows()
	c.release()

if __name__ == "__main__":
	show_video(sys.argv[1:])
