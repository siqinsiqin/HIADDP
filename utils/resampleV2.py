# -*-coding:utf-8 -*-
"""
# Time       ：2023/3/2 13:22
# Author     ：comi
# version    ：python 3.8
# Description： resample 多名医生的均值掩码
"""
import numpy as np
from scipy import ndimage

# For contour to boolean mask function.

# For CT volume visualizer.

# For diameter estimation.
from scipy.interpolate import RegularGridInterpolator

# For 3D visualizer.

try:
    from skimage.measure import marching_cubes
except ImportError:
    # Old version compatible since marching_cubes replaced with marchin_cubes_lewiner in skimage 0.19.0
    from skimage.measure import marching_cubes_lewiner as marching_cubes


def uniform_cubic_resample(side_length=None, bbox_dims=None,
                           bbox=None, pixel_spacing=None, slice_spacing=None, slice_zvals=None,
                           mask=None, scan=None, resample_vol=True,
                           irp_pts=None, return_irp_pts=False, verbose=False):
    bbox = np.array([[sl.start, sl.stop - 1] for sl in bbox])
    bboxd = bbox_dims
    rij = pixel_spacing
    rk = slice_spacing

    imin, imax = bbox[0]
    jmin, jmax = bbox[1]
    kmin, kmax = bbox[2]

    xmin, xmax = imin * rij, imax * rij
    ymin, ymax = jmin * rij, jmax * rij

    zmin = slice_zvals[kmin]
    zmax = slice_zvals[kmax]

    # { Begin input checks.
    if side_length is None:
        side_length = np.ceil(bboxd.max())
    else:
        if not isinstance(side_length, int):
            raise TypeError('`side_length` must be an integer.')
        if side_length < bboxd.max():
            raise ValueError('`side_length` must be greater\
                               than any bounding box dimension.')
    side_length = float(side_length)
    # } End input checks.

    # Load the images. Get the z positions.
    images = scan.load_all_dicom_images(verbose=verbose)
    img_zs = [float(img.ImagePositionPatient[-1]) for img in images]
    img_zs = np.unique(img_zs)

    # Get the z values of the contours.
    # contour_zs = np.unique([c.image_z_position for c in self.contours])

    # Get the indices where the nodule stops and starts
    # with respect to the scan z values.
    # kmin = np.where(zmin == img_zs)[0][0]
    # kmax = np.where(zmax == img_zs)[0][0]

    # Initialize the boolean mask.
    # mask = self.boolean_mask()

    ########################################################
    # { Begin interpolation grid creation.
    #   (The points at which the volumes will be resampled.)

    # Compute new interpolation grid points in x.
    d = 0.5 * (side_length - (xmax - xmin))
    xhat, step = np.linspace(xmin - d, xmax + d,
                             int(side_length) + 1, retstep=True)
    assert (abs(step - 1) < 1e-5).all(), "New x spacing != 1."

    # Do the same for y.
    d = 0.5 * (side_length - (ymax - ymin))
    yhat, step = np.linspace(ymin - d, ymax + d,
                             int(side_length) + 1, retstep=True)
    assert (abs(step - 1) < 1e-5).all(), "New y spacing != 1."

    # Do the same for z.
    d = 0.5 * (side_length - (zmax - zmin))
    zhat, step = np.linspace(zmin - d, zmax + d,
                             int(side_length) + 1, retstep=True)
    assert (abs(step - 1) < 1e-5).all(), "New z pixel spacing != 1."

    # } End interpolation grid creation.
    ########################################################

    ########################################################
    # { Begin grid creation.
    #   (The points at which the volumes are assumed to be sampled.)

    # a[x|y|z], b[x|y|z] are the start / stop indexes for the
    # (non resample) sample grids along each respective axis.

    # It helps to draw a diagram. For example,
    #
    # *--*--*-- ...
    # x3 x4 x5
    #  *---*---*--- ...
    #  xhat0
    #
    # In this case, `ax` would be chosen to be 3
    # since this is the index directly to the left of
    # `xhat[0]`. If `xhat[0]` is below any grid point,
    # then `ax` is the minimum possible index, 0. A similar
    # diagram helps with the `bx` index.

    T = np.arange(0, 512) * rij

    if xhat[0] <= 0:
        ax = 0
    else:
        ax = (T < xhat[0]).sum() - 1
    if xhat[-1] >= T[-1]:
        bx = 512
    else:
        bx = 512 - (T > xhat[-1]).sum() + 1

    if yhat[0] <= 0:
        ay = 0
    else:
        ay = (T < yhat[0]).sum() - 1
    if yhat[-1] >= T[-1]:
        by = 512
    else:
        by = 512 - (T > yhat[-1]).sum() + 1

    if zhat[0] <= img_zs[0]:
        az = 0
    else:
        az = (img_zs < zhat[0]).sum() - 1
    if zhat[-1] >= img_zs[-1]:
        bz = len(img_zs)
    else:
        bz = len(img_zs) - (img_zs > zhat[-1]).sum() + 1

    # These are the actual grids.
    x = T[ax:bx]
    y = T[ay:by]
    z = img_zs[az:bz]

    # } End grid creation.
    ########################################################

    # Create the non-interpolated CT volume.
    if resample_vol:
        ctvol = np.zeros(x.shape + y.shape + z.shape, dtype=np.float64)
        for k in range(z.shape[0]):
            ctvol[:, :, k] = images[k + az].pixel_array[ax:bx, ay:by]

    # We currently only have the boolean mask volume on the domain
    # of the bounding box. Thus, we must "place it" in the appropriately
    # sized volume (i.e., `ctvol.shape`). This is accomplished by
    # padding `mask`.
    padvals = [(imin - ax, bx - 1 - imax),  # The `b` terms have a `+1` offset
               (jmin - ay, by - 1 - jmax),  # from being an index that is
               (kmin - az, bz - 1 - kmax)]  # corrected with the `-1` here.
    mask = np.pad(mask, pad_width=padvals,
                  mode='constant', constant_values=False)

    # Obtain minimum image value to use as const for interpolation.
    if resample_vol:
        fillval = min([img.pixel_array.min() for img in images])

    if irp_pts is None:
        ix, iy, iz = np.meshgrid(xhat, yhat, zhat, indexing='ij')
    else:
        ix, iy, iz = irp_pts
    IXYZ = np.c_[ix.flatten(), iy.flatten(), iz.flatten()]

    # Interpolate the nodule CT volume.
    if resample_vol:
        rgi = RegularGridInterpolator(points=(x, y, z), values=ctvol,
                                      bounds_error=False, fill_value=fillval)
        ictvol = rgi(IXYZ).reshape(ix.shape)

    # Interpolate the mask volume.
    rgi = RegularGridInterpolator(points=(x, y, z), values=mask,
                                  bounds_error=False, fill_value=False)
    imask = rgi(IXYZ).reshape(ix.shape) > 0

    if resample_vol:
        if return_irp_pts:
            return ictvol, imask, (ix, iy, iz)
        else:
            return ictvol, imask
    else:
        if return_irp_pts:
            return imask, (ix, iy, iz)
        else:
            return imask


from skimage.transform import resize


def rescale_image(image, original_spacing, target_spacing, target_size):
    # Compute the resize factor based on the original and target spacings
    resize_factor = np.array(original_spacing) / np.array(target_spacing)

    # Compute the new shape of the image after resampling
    new_shape = np.round(image.shape * resize_factor)

    # Resample the image to the new shape using cubic interpolation
    resampled_image = ndimage.zoom(image, resize_factor, order=3)

    # Resize the resampled image to the target size using trilinear interpolation
    rescaled_image = resize(resampled_image, target_size, order=1, mode='reflect', anti_aliasing=True)

    return rescaled_image


def rescale_mask(mask, original_spacing, target_spacing, target_size):
    # Compute the resize factor based on the original and target spacings
    resize_factor = np.array(original_spacing) / np.array(target_spacing)

    # Compute the new shape of the mask after resampling
    new_shape = np.round(mask.shape * resize_factor)

    # Resample the mask to the new shape using nearest-neighbor interpolation
    resampled_mask = ndimage.zoom(mask, resize_factor, order=0)

    # Resize the resampled mask to the target size using nearest-neighbor interpolation
    rescaled_mask = resize(resampled_mask, target_size, order=0, mode='reflect', anti_aliasing=False)

    return rescaled_mask
