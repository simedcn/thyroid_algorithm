import matplotlib.pyplot as plt
import pydicom
import cv2
from pydicom.data import get_testdata_files

print(__doc__)
filename="C:\\Users\\lenovo\\Desktop\\1.dcm"
#filename = get_testdata_files('CT_small.dcm')[0]
dataset = pydicom.dcmread(filename)

# Normal mode:
print()
print("Filename.........:", filename)
print("Storage type.....:", dataset.SOPClassUID)
print()

pat_name = dataset.PatientName
display_name = pat_name.family_name + ", " + pat_name.given_name
print("Patient's name...:", display_name)
print("Patient id.......:", dataset.PatientID)
print("Modality.........:", dataset.Modality)
print("Study Date.......:", dataset.StudyDate)

if 'PixelData' in dataset:
    rows = int(dataset.Rows)
    cols = int(dataset.Columns)
    print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
        rows=rows, cols=cols, size=len(dataset.PixelData)))
    if 'PixelSpacing' in dataset:
        print("Pixel spacing....:", dataset.PixelSpacing)

# use .get() if not sure the item exists, and want a default value if missing
print("Slice location...:", dataset.get('SliceLocation', "(missing)"))

# plot the image using matplotlib
plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
cv2.imwrite("C:\\Users\\lenovo\\Desktop\\cv.jpg",dataset.pixel_array)
#plt.show()
#plt.savefig("C:\\Users\\lenovo\\Desktop\\1.png")
