import glob
import scipy.io
import cv2
import numpy as np

pixel_level_files = glob.glob("./annotations/image-level/*.mat")

for x in pixel_level_files:
    mat = scipy.io.loadmat(x)
    mat = np.array(mat['groundtruth'])
    # cv2.imshow("img", mat)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite("./mask/{0}".format(x.split("/")[-1].replace(".mat", ".jpg")), mat)