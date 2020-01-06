import os
import numpy as np
import SimpleITK as itk
from tqdm import tqdm
from keras import backend as K
import matplotlib.pyplot as plt

from os.path import join
dimension = 2
def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = itk.sitkNearestNeighbor
    default_value = 0
    return itk.Resample(image, reference_image, transform,
                         interpolator, default_value)

def get_center(img):
    """
    This function returns the physical center point of a 3d sitk image
    :param img: The sitk image we are trying to find the center of
    :return: The physical center point of the image
    """
    width, height = img.GetSize()
    return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                              int(np.ceil(height/2))))
def affine_rotate_euler(img, degrees=0):
    img_center = get_center(img)
    radians = np.deg2rad(degrees)
    #affine = itk.Euler2DTransform()
    #affine = itk.VersorTransform((0,0,1), radians)
    affine = itk.Similarity2DTransform();
    affine.SetAngle(radians)
    affine.SetCenter(img_center)
    new_transform = affine
    return new_transform

def affine_scale(img, x_scale=1, y_scale=1):
    img_center = get_center(img)
    affine = itk.ScaleTransform(2, (x_scale, y_scale))
    affine.SetCenter(img_center)
    new_transform = affine
    return new_transform

def random_rotate_scale(input_image,degrees=0,x_scale=1,y_scale=1):
    affine = itk.Transform(affine_rotate_euler(input_image,degrees=degrees))
    affine.AddTransform(itk.Transform(affine_scale(input_image,x_scale = x_scale, y_scale = y_scale)))
    
    output_image = resample(input_image, affine)
    return output_image


def resample_image(input_image,output_size, spacing_mm=(1, 1), spacing_image=None, inter_type=itk.sitkBSpline):
    """
    Resample image to desired pixel spacing.

    Should specify destination spacing immediate value in parameter spacing_mm or as SimpleITK.Image in spacing_image.
    You must specify either spacing_mm or spacing_image, not both at the same time.

    :param input_image: Image to resample.
    :param spacing_mm: Spacing for resampling in mm given as tuple or list of two/three (2D/3D) float values.
    :param spacing_image: Spacing for resampling taken from the given SimpleITK.Image.
    :param inter_type: Interpolation type using one of the following options:
                            SimpleITK.sitkNearestNeighbor,
                            SimpleITK.sitkLinear,
                            SimpleITK.sitkBSpline,
                            SimpleITK.sitkGaussian,
                            SimpleITK.sitkLabelGaussian,
                            SimpleITK.sitkHammingWindowedSinc,
                            SimpleITK.sitkBlackmanWindowedSinc,
                            SimpleITK.sitkCosineWindowedSinc,
                            SimpleITK.sitkWelchWindowedSinc,
                            SimpleITK.sitkLanczosWindowedSinc
    :type input_image: SimpleITK.Image
    :type spacing_mm: Tuple[float]
    :type spacing_image: SimpleITK.Image
    :type inter_type: int
    :rtype: SimpleITK.Image
    :return: Resampled image as SimpleITK.Image.
    """
    resampler = itk.ResampleImageFilter()
    resampler.SetInterpolator(inter_type)

    if (spacing_mm is None and spacing_image is None) or \
       (spacing_mm is not None and spacing_image is not None):
        raise ValueError('You must specify either spacing_mm or spacing_image, not both at the same time.')

    if spacing_image is not None:
        spacing_mm = spacing_image.GetSpacing()

    input_spacing = input_image.GetSpacing()
    #print(input_spacing)
    # set desired spacing
    resampler.SetOutputSpacing(spacing_mm)
    # compute and set output size
    #output_size = np.array(input_image.GetSize()) * np.array(input_spacing) \
    #             / np.array(spacing_mm)
    
    #print(output_size)
    output_size = np.array(output_size)
    output_size = list((output_size).astype('uint32'))
    output_size = [int(size) for size in output_size]
    resampler.SetSize(output_size)
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())
    resampled_image = resampler.Execute(input_image)

    return resampled_image




def load_mri_brain_data(output_size=(64, 64), modalities=('t1'), data_location='./data'):
    """
    Load data from '../mri-brain-slices' folder in the format suitable for training neural networks.

    This functions will load all images, perform cropping to fixed size and resample the obtained image such
    that the output size will match the one specified by parameter 'output_size'. The data output will contain the
    modalities as specified by parameter 'modalities'.

    :param output_size: Define output image size.
    :param modalities: List of modalities specified by strings 't1' and/or 'flair'.
    :type output_size: tuple[int]
    :type modalities: tuple[str]
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :return: Image data, brainmask and brain segmentation in a tuple.
    """
    # hidden function parameters
   
 # DATA_PATH = './data_t1_hires'
    DATA_PATH = data_location
    # define image extraction function based on cropping and resampling
    def extract_image(image, output_size=(64, 64), interpolation_type=itk.sitkGaussian, slice_number=1):
        new_spacing_mm = (image.GetSpacing()[1]*(image.GetSize()[1] / output_size[0]),image.GetSpacing()[2]* (image.GetSize()[2] / output_size[1]))
        # extract slice
        index = [slice_number,0,0]
        size = list(image.GetSize())
        size[0] = 0
        Extractor = itk.ExtractImageFilter()
        Extractor.SetSize(size)
        Extractor.SetIndex(index)
        image_slice = Extractor.Execute(image)
        return resample_image(
            itk.RegionOfInterest(image_slice, (image.GetSize()[1], image.GetSize()[2]), (0, 0)), 
            output_size,
            spacing_mm = new_spacing_mm, 
            inter_type=interpolation_type)
    
    # load and extract all images and masks into a list of dicts
    mri_data = []
    patient_paths = os.listdir(DATA_PATH)
    if '.ipynb_checkpoints' in patient_paths:
        patient_paths.remove('.ipynb_checkpoints')
    
    for pacient_no in tqdm(range(len(patient_paths))):
        patient_path = join(DATA_PATH, patient_paths[pacient_no])

        # read all images
      
        seg1 = itk.ReadImage(join(patient_path, 'seg.nrrd'))
        t1 = itk.ReadImage(join(patient_path, 't1.nrrd'))
       
        for slice_number in range(0,t1.GetSize()[0]):
        # crop and resample the images
            seg1_extracted = extract_image(seg1, output_size, itk.sitkNearestNeighbor, slice_number)
            t1_extracted = extract_image(t1, output_size, itk.sitkNearestNeighbor, slice_number)
        
            
            
            mri_data.append({'t1':t1_extracted, 'seg1':seg1_extracted})   
        
    # reshape all modalities and masks into 3d arrays
   
    t1_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['t1'])) for data in mri_data])
    seg1_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['seg1'])) for data in mri_data])
   
    
    # reshape the 3d arrays such that the number of cases is in the first column
   
    t1_array = np.transpose(t1_array, (2, 0, 1))
    seg1_array = np.transpose(seg1_array, (2, 0, 1))
   
    # reshape the 3d arrays according to the Keras backend
    if K.image_data_format() == 'channels_first': 
        # this format is (n_cases, n_channels, image_height, image_width)
       
        t1_karray = t1_array[:, np.newaxis, :, :]
        seg1_karray = seg1_array[:, np.newaxis, :, :]
        channel_axis = 1
    else:
        # this format is (n_cases, image_height, image_width, n_channels)
      
        t1_karray = t1_array[:, :, :, np.newaxis]
        seg1_karray = seg1_array[:, :, :, np.newaxis]
  
        channel_axis = -1

    
    data = t1_karray        
    # read image sizes and channel number
    _, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = data.shape

    # compute min and max values per each channel
    def stat_per_channel(values, stat_fcn):
        return stat_fcn(
            np.reshape(
                values, 
                (values.shape[0]*IMG_HEIGHT*IMG_WIDTH, IMG_CHANNELS)), 
            axis=0)[:, np.newaxis]

    min_data, max_data = stat_per_channel(data, np.min), stat_per_channel(data, np.max)
    min_data = np.reshape(min_data, (1, 1, 1, IMG_CHANNELS))
    max_data = np.reshape(max_data, (1, 1, 1, IMG_CHANNELS))

    # normalize image intensities to interval [0, 1]
    X = (data - min_data) / (max_data - min_data)
    
    # return the image modalities, brainmasks and brain segmentations
    return X, seg1_karray


def load_mri_brain_data_3slices(output_size=(64, 64), modalities=('t1'), data_location='./data'):
    """
    Load data from '../mri-brain-slices' folder in the format suitable for training neural networks.

    This functions will load all images, perform cropping to fixed size and resample the obtained image such
    that the output size will match the one specified by parameter 'output_size'. The data output will contain the
    modalities as specified by parameter 'modalities'.

    :param output_size: Define output image size.
    :param modalities: List of modalities specified by strings 't1' and/or 'flair'.
    :type output_size: tuple[int]
    :type modalities: tuple[str]
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :return: Image data, brainmask and brain segmentation in a tuple.
    """
    # hidden function parameters
   
 # DATA_PATH = './data_t1_hires'
    DATA_PATH = data_location
    # define image extraction function based on cropping and resampling
    def extract_image(image, output_size=(64, 64), interpolation_type=itk.sitkGaussian, slice_number=1):
        new_spacing_mm = (image.GetSpacing()[1]*(image.GetSize()[1] / output_size[0]),image.GetSpacing()[2]* (image.GetSize()[2] / output_size[1]))
        # extract slice
        index = [slice_number,0,0]
        size = list(image.GetSize())
        size[0] = 0
        Extractor = itk.ExtractImageFilter()
        Extractor.SetSize(size)
        Extractor.SetIndex(index)
        image_slice = Extractor.Execute(image)
        return resample_image(
            itk.RegionOfInterest(image_slice, (image.GetSize()[1], image.GetSize()[2]), (0, 0)), 
            output_size,
            spacing_mm = new_spacing_mm, 
            inter_type=interpolation_type)
    
    # load and extract all images and masks into a list of dicts
    mri_data = []
    patient_paths = os.listdir(DATA_PATH)
    if '.ipynb_checkpoints' in patient_paths:
        patient_paths.remove('.ipynb_checkpoints')
    
    for pacient_no in tqdm(range(len(patient_paths))):
        patient_path = join(DATA_PATH, patient_paths[pacient_no])

        # read all images
      
        seg1 = itk.ReadImage(join(patient_path, 'seg.nrrd'))
        t1 = itk.ReadImage(join(patient_path, 't1.nrrd'))
       
        for slice_number in range(0,t1.GetSize()[0]):
        # crop and resample the images
            seg1_extracted = extract_image(seg1, output_size, itk.sitkNearestNeighbor, slice_number)
            
            previous_slice_number = slice_number - 1
            next_slice_number = slice_number + 1
            
            if ((slice_number - 1) < 0):
                previous_slice_number = slice_number
            if ((slice_number) >  (t1.GetSize()[0] - 2)):
                next_slice_number = slice_number
          
            
            
            t1_extracted_previous_slice = extract_image(t1, output_size, itk.sitkNearestNeighbor, previous_slice_number)
            t1_extracted_middle_slice = extract_image(t1, output_size, itk.sitkNearestNeighbor, slice_number)
            t1_extracted_next_slice = extract_image(t1, output_size, itk.sitkNearestNeighbor, next_slice_number)
            
            #generate random data transform max +/- 10 deg rotation, scale +/- 5%
            degrees = np.random.uniform(-10,10)
            x_scale = np.random.uniform(0.95,1.05)
            y_scale = np.random.uniform(0.95,1.05)
            
            #transform all slices and segmentation
            seg1_extracted = random_rotate_scale(seg1_extracted,degrees,x_scale,y_scale )
            t1_extracted_previous_slice = random_rotate_scale(t1_extracted_previous_slice,degrees,x_scale,y_scale)
            t1_extracted_middle_slice = random_rotate_scale(t1_extracted_middle_slice,degrees,x_scale,y_scale)
            t1_extracted_next_slice = random_rotate_scale(t1_extracted_next_slice,degrees,x_scale,y_scale)
            
            
            mri_data.append({'t11':t1_extracted_previous_slice,'t12':t1_extracted_middle_slice,'t13':t1_extracted_next_slice, 'seg1':seg1_extracted})   
        
    # reshape all modalities and masks into 3d arrays
   
    t1_array1 = np.dstack([np.squeeze(itk.GetArrayFromImage(data['t11'])) for data in mri_data])
    t1_array2 = np.dstack([np.squeeze(itk.GetArrayFromImage(data['t12'])) for data in mri_data])
    t1_array3 = np.dstack([np.squeeze(itk.GetArrayFromImage(data['t13'])) for data in mri_data])
    seg1_array = np.dstack([np.squeeze(itk.GetArrayFromImage(data['seg1'])) for data in mri_data])
   
    
    # reshape the 3d arrays such that the number of cases is in the first column
   
    t1_array1 = np.transpose(t1_array1, (2, 0, 1))
    t1_array2 = np.transpose(t1_array2, (2, 0, 1))
    t1_array3 = np.transpose(t1_array3, (2, 0, 1))
    seg1_array = np.transpose(seg1_array, (2, 0, 1))
   
    # reshape the 3d arrays according to the Keras backend
    if K.image_data_format() == 'channels_first': 
        # this format is (n_cases, n_channels, image_height, image_width)
       
        t1_karray1 = t1_array1[:, np.newaxis, :, :]
        t1_karray2 = t1_array2[:, np.newaxis, :, :]
        t1_karray3 = t1_array3[:, np.newaxis, :, :]
        seg1_karray = seg1_array[:, np.newaxis, :, :]
        channel_axis = 1
    else:
        # this format is (n_cases, image_height, image_width, n_channels)
      
        t1_karray1 = t1_array1[:, :, :, np.newaxis]
        t1_karray2 = t1_array2[:, :, :, np.newaxis]
        t1_karray3 = t1_array3[:, :, :, np.newaxis]
        seg1_karray = seg1_array[:, :, :, np.newaxis]
        channel_axis = -1

    
    data = np.concatenate((t1_karray1,t1_karray2,t1_karray3),axis=channel_axis)        
    # read image sizes and channel number
    _, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = data.shape

    # compute min and max values per each channel
    def stat_per_channel(values, stat_fcn):
        return stat_fcn(
            np.reshape(
                values, 
                (values.shape[0]*IMG_HEIGHT*IMG_WIDTH, IMG_CHANNELS)), 
            axis=0)[:, np.newaxis]

    min_data, max_data = stat_per_channel(data, np.min), stat_per_channel(data, np.max)
    min_data = np.reshape(min_data, (1, 1, 1, IMG_CHANNELS))
    max_data = np.reshape(max_data, (1, 1, 1, IMG_CHANNELS))

    # normalize image intensities to interval [0, 1]
    X = (data - min_data) / (max_data - min_data)
    
    # return the image modalities, brainmasks and brain segmentations
    return X, seg1_karray


