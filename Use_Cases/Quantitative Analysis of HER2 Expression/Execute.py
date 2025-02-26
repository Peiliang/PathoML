import numpy as np
from skimage.measure import label, regionprops
import cv2
import sqlite3 as sql
import subprocess

conn = sql.connect('Annotation.db')
fout = open('Property.csv', 'w')
c = conn.cursor()

def process(id, points):
	mnx, mny, mxx, mxy = 1e6, 1e6, 0, 0
	contour = []
	for row in points:
		mnx = min(row[1] - 1, mnx)
		mny = min(row[2] - 1, mny)
		mxx = max(row[1] + 2, mxx)
		mxy = max(row[2] + 2, mxy)
		contour.append([row[1], row[2]])
	contour.append([points[-1][3], points[-1][4]])

	fp, lp = contour[0], contour[-1]
	closed = (fp[0] - lp[0]) ** 2 + (fp[1] - lp[1]) ** 2 <= 10
	ctx, cty, cw = 0, 0, 0

	w, h = mxx - mnx, mxy - mny
	for point in contour:
		point[0] -= mnx
		point[1] -= mny
	img = np.zeros((h, w), dtype=np.uint8)
	contour = np.array(contour, dtype=np.int32)
	cv2.fillPoly(img, [contour], 255)

	label_img = label(img == 255)
	region = regionprops(label_img)[0]
	ctx, cty = region.centroid[1], region.centroid[0]
	cw, ccw = 0, 0
	for i in range(len(contour) - 1):
		x1, y1, x2, y2 = contour[i][0] - ctx, contour[i][1] - cty, contour[i + 1][0] - ctx, contour[i + 1][1] - cty
		cross = x1 * y2 - x2 * y1
		if cross > 0:
			cw += 1
		else:
			ccw += 1
	fout.write('%d,%d,%d,%f,%f,%d,%d,%d,%d,%d\n' % (id, mnx, mny, ctx, cty, fp[0], fp[1], lp[0], lp[1], cw > ccw))

data = c.execute('SELECT * from Line')
points = []
id = 1
for row in data:
	if row[6] != id:
		process(id, points)
		id += 1
		points.clear()
	points.append(row)
process(id, points)

conn.close()
fout.close()

subprocess.call(['java', '-jar', 'CalculateHER2.jar'])
