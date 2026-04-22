# -*-coding:utf-8 -*-
"""
# Time       ：2022/9/20 21:00
# Author     ：comi
# version    ：python 3.8
# Description：
todo 挑选几种较为常见的：GGO,粘连型,空洞等，描述几位医生不同的表述
图画+表格

粘连：759-0 √
GGO: 439-2
Moderately Subtle 2 Soft Tissue 1 Absent 6 Ovoid 3 Medium Margin 3 No Lobulation 1 No Spiculation 1 Solid/Mixed 4 Highly Unlikely 1
Moderately Obvious 4 Soft Tissue 1 Absent 6 Ovoid 3 Near Poorly Defined 2 Medium Lobulation 3 No Spiculation 1 Non-Solid/Mixed 2 Indeterminate 3
Fairly Subtle 3 Soft Tissue 1 Absent 6 Round 5 Medium Margin 3 Medium Lobulation 3 No Spiculation 1 Non-Solid/GGO 1 Indeterminate 3

空洞： 644-0 √

solid: 182-2
Obvious 5 Soft Tissue 1 Absent 6 Ovoid/Round 4 Near Sharp 4 Near Marked Lobulation 4 Nearly No Spiculation 2 Solid 5 Indeterminate 3
Fairly Subtle 3 Soft Tissue 1 Absent 6 Round 5 Sharp 5 No Lobulation 1 No Spiculation 1 Solid 5 Moderately Unlikely 2
Obvious 5 Soft Tissue 1 Absent 6 Ovoid 3 Near Sharp 4 No Lobulation 1 No Spiculation 1 Solid/Mixed 4 Moderately Suspicious 4
Obvious 5 Soft Tissue 1 Absent 6 Ovoid/Round 4 Sharp 5 Medium Lobulation 3 No Spiculation 1 Solid 5 Highly Suspicious 5
钙化：124-0
Obvious 5 Soft Tissue 1 Laminated 2 Ovoid 3 Sharp 5 No Lobulation 1 Medium Spiculation 3 Solid 5 Highly Unlikely 1
Obvious 5 Soft Tissue 1 Laminated 2 Round 5 Sharp 5 No Lobulation 1 No Spiculation 1 Solid 5 Highly Unlikely 1
Obvious 5 Soft Tissue 1 Solid 3 Ovoid 3 Sharp 5 No Lobulation 1 No Spiculation 1 Solid 5 Highly Unlikely 1
Obvious 5 Soft Tissue 1 Solid 3 Round 5 Sharp 5 No Lobulation 1 No Spiculation 1 Solid 5 Highly Unlikely 1
尖刺：2-2，18-0, final 34-0
Obvious 5 Soft Tissue 1 Absent 6 Ovoid 3 Medium Margin 3 Medium Lobulation 3 Marked Spiculation 5 Solid 5 Highly Suspicious 5
Obvious 5 Soft Tissue 1 Absent 6 Ovoid 3 Near Sharp 4 Near Marked Lobulation 4 Marked Spiculation 5 Solid 5 Highly Suspicious 5
Obvious 5 Soft Tissue 1 Absent 6 Ovoid/Round 4 Near Poorly Defined 2 Near Marked Lobulation 4 Marked Spiculation 5 Solid 5 Highly Suspicious 5
Obvious 5 Soft Tissue 1 Absent 6 Round 5 Near Sharp 4 No Lobulation 1 Marked Spiculation 5 Solid/Mixed 4 Moderately Suspicious 4
恶性：1-3
Obvious 5 Soft Tissue 1 Non-central 4 Ovoid 3 Sharp 5 Nearly No Lobulation 2 Medium Spiculation 3 Solid 5 Moderately Suspicious 4
Obvious 5 Soft Tissue 1 Absent 6 Ovoid/Round 4 Near Poorly Defined 2 Near Marked Lobulation 4 No Spiculation 1 Solid 5 Highly Suspicious 5
Obvious 5 Soft Tissue 1 Non-central 4 Ovoid/Round 4 Near Sharp 4 Near Marked Lobulation 4 Near Marked Spiculation 4 Solid 5 Highly Suspicious 5
Obvious 5 Soft Tissue 1 Absent 6 Ovoid/Round 4 Near Poorly Defined 2 Medium Lobulation 3 Medium Spiculation 3 Solid/Mixed 4 Indeterminate 3

细微：969-7
Fairly Subtle 3 Soft Tissue 1 Absent 6 Round 5 Poorly Defined 1 Medium Lobulation 3 No Spiculation 1 Non-Solid/GGO 1 Indeterminate 3
Fairly Subtle 3 Soft Tissue 1 Absent 6 Ovoid/Round 4 Poorly Defined 1 No Lobulation 1 No Spiculation 1 Non-Solid/Mixed 2 Moderately Suspicious 4
Moderately Obvious 4 Soft Tissue 1 Absent 6 Ovoid/Round 4 Near Poorly Defined 2 Nearly No Lobulation 2 Nearly No Spiculation 2 Non-Solid/GGO 1 Indeterminate 3

分叶：953-4
Obvious 5 Soft Tissue 1 Absent 6 Ovoid/Linear 2 Medium Margin 3 Medium Lobulation 3 Medium Spiculation 3 Part Solid/Mixed 3 Moderately Suspicious 4
Obvious 5 Soft Tissue 1 Absent 6 Ovoid/Round 4 Near Poorly Defined 2 Near Marked Lobulation 4 Nearly No Spiculation 2 Part Solid/Mixed 3 Highly Suspicious 5
Obvious 5 Soft Tissue 1 Absent 6 Ovoid 3 Medium Margin 3 Marked Lobulation 5 Marked Spiculation 5 Solid/Mixed 4 Highly Suspicious 5

"""
import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl
from pylidc.utils import consensus

# Query for a scan, and convert it to an array volume.
from utils.helper import lumTrans
CTs = pl.query(pl.Annotation).all()

scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == 'LIDC-IDRI-0001').first()  # 0255 ，0635
# scan = pl.query(pl.Scan).filter(pl.Scan.study_instance_uid == 1).first()  # 969,644


print(scan)
vol = scan.to_volume()

# Cluster the annotations for the scan, and grab one.
nods = scan.cluster_annotations()
print(nods)


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


# for nod in nods:
anns = nods[0]

# Perform a consensus consolidation and 50% agreement level.
# We pad the slices to add context for viewing.
for ann in anns:
    print(
        ann.Subtlety, ann.subtlety,
        ann.InternalStructure, ann.internalStructure,
        ann.Calcification, ann.calcification,
        ann.Sphericity, ann.sphericity,
        ann.Margin, ann.margin,
        ann.Lobulation, ann.lobulation,
        ann.Spiculation, ann.spiculation,
        ann.Texture, ann.texture,
        ann.Malignancy, ann.malignancy,
    )
    voxelcoord = worldToVoxelCoord([], scan.origin, scan.spacings)

size = 30
cmask, cbbox, masks = consensus(anns, clevel=0.5, pad=[(size, size), (size, size), (0, 0)])

vol = lumTrans(vol)
# Get the central slice of the computed bounding box.
k = int(0.5 * (cbbox[2].stop - cbbox[2].start))

# Set up the plot.
fig, ax = plt.subplots(1, 1, figsize=(5, 5))


#
# an_filter = anisodiff2D()
# result = MedianFilter(vol[cbbox][:, :, k], k=2, padding=0)
# ax.imshow(result, cmap=plt.cm.gray, alpha=1)

# diff_im = an_filter.fit(vol[cbbox][:, :, k])
# diff_im = an_filter.fit(result)
# ax.imshow(diff_im, cmap=plt.cm.gray, alpha=1)  #
def cut_norm(img):
    # 截断归一化
    lungwin = np.array([-1000., 400.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return newimg


img = cut_norm(vol[cbbox])

ax.imshow(img[:, :, k], cmap=plt.cm.gray, alpha=1, interpolation='sinc', )  # lanczos,spline36,catrom
# ax.imshow(masks[3][:, :, k], cmap=plt.cm.gray, alpha=1, interpolation='sinc', )  # lanczos,spline36,catrom
# ax.imshow(cmask[:, :, k], cmap=plt.cm.gray, alpha=1, interpolation='sinc', )  # lanczos,spline36,catrom

# img = sobel_filter(img)[0]
# ax.imshow(img[:, :, k], cmap=plt.cm.gray, alpha=1, interpolation='sinc', )  # lanczos,spline36,catrom

# Plot the annotation contours for the kth slice.
colors = ['r', '#8A2BE2', '#33A139', '#FF9912', 'c']

# for j in range(len(masks)):
#     for c in find_contours(masks[j][:, :, k].astype(float), 0.5):
#         label = "Annotation %d" % (j + 1)
#         plt.plot(c[:, 1], c[:, 0], colors[j], label=label, linewidth=2)
#
# # Plot the 50% consensus contour for the kth slice.
# for c in find_contours(cmask[:, :, k].astype(float), 0.5):
#     plt.plot(c[:, 1], c[:, 0], '--b', label='50% Consensus', linewidth=3)

ax.axis('off')
# ax.legend()
plt.tight_layout()
plt.savefig("../holl.png", bbox_inches="tight", pad_inches=0)
plt.show()
