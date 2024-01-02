import SimpleITK as sitk
from skimage import transform
import numpy as np
import warnings
import pywt
import cv2

warnings.filterwarnings('ignore')
def mhatonii(case,num):
    # img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mha\Creates/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")
    # sitk.WriteImage(img, r"D:\4DCT\Data\merge\cropwhole\nii\Creates/case" + str(case) + "/case" + str(case) + "_" + num + ".nii")
    #
    img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mha\NLST/case" + str(case) + "_" + num + ".mha")
    sitk.WriteImage(img, r"D:\4DCT\Data\merge\cropwhole\nii\NLST/case" + str(case) + "_" + num + ".nii")
def niitomha(case,num):
    img = sitk.ReadImage("E:/data/mhalungairenhance/" + case + "/" + case + "_" + num + "vessel.nii.gz")
    sitk.WriteImage(img, "E:/data/mhalungairenhance/" + case + "/" + case + "_" + num + "vessel.mha")
def npytomha(case,num):
    arr = np.load(r"D:\4DCT\Data\merge\Enhance615_400_-200-1000_01\none\Spare/case" + str(case) + "_" + num + ".npy")
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img,r"D:\4DCT\Data\merge\Enhance615_400_-200-1000_01\mha\Spare\case" + str(case) + "_" + num + ".mha")
def mhatonpy(case,num):
    img = sitk.ReadImage(r"D:\4DCT\Data\merge\Enhance615_400_-200-1000_01FFD\FFD5\Spare/case" + str(case) + "_" + num + ".mha")
    arr = sitk.GetArrayFromImage(img)
    np.save(r"D:\4DCT\Data\merge\Enhance615_400_-200-1000_01FFD\FFD5\Spare/case" + str(case) + "_" + num + ".npy", arr)
def npytonii(case,num):
    arr = np.load(r"E:\data\SPARE\SpareOrigin01/case" + str(case) + "/case" +str(case) + "_" + num + ".npy")
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img,r"E:\data\SPARE\SpareEnhance01\case" + str(case) + "/case" +str(case) + "_" + num + ".nii")
def Close(case,num):
    # img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\seg\Spare\case" + str(case) + "/case" + str(case) + "_" + num + ".mha")
    img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\seg\NLST/case" + str(case) + "_" + num + ".mha")
    # imgopen = sitk.GrayscaleMorphologicalOpening(img)
    radius = 1
    di = sitk.GrayscaleDilateImageFilter()
    di.SetKernelType(sitk.sitkBall)
    di.SetKernelRadius(radius)
    img_di = di.Execute(img)

    er = sitk.GrayscaleErodeImageFilter()
    er.SetKernelType(sitk.sitkBall)
    er.SetKernelRadius(radius)
    img_er = er.Execute(img_di)
    imgclose = img_er

    sitk.WriteImage(imgclose, r"D:\4DCT\Data\merge\cropwhole\seg\NLST/case" + str(case) + "_" + num + "segclose.nii")
    # sitk.WriteImage(imgopen, r"D:\4DCT\Data\merge\cropwhole\seg\Spare\case" + str(case) + "/case" + str(case) + "_" + num + "segclose.nii")

def ImageResample(case, num):
    target_img = sitk.ReadImage("E:\data/mha/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")
    # target_img = sitk.ReadImage("E:\data/mha/case" + str(case) + "/case" + str(case) + "_" + num + "Resample.mha")
    # target_Size = target_img.GetSize()
    target_Spacing = (1.0, 1.0, 2.5)
    target_origin = target_img.GetOrigin()
    target_direction = target_img.GetDirection()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target_img)
    # resampler.SetSize(target_Size)
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkBSplineResamplerOrder4)
    sitk_image_resampled = resampler.Execute(target_img)
    sitk.WriteImage(sitk_image_resampled,
                    "E:\data/mha/case" + str(case) + "/case" + str(case) + "_" + num + "Resample.mha")
def switchtoinhao(case,num):
    img_flip = sitk.ReadImage("E:\data\mhacropnew/case" + str(case) + "/case" + str(case) + "_" + num + "fusion.mha")
    img_arr = sitk.GetArrayFromImage(img_flip)
    img_arr = transform.resize(img_arr, (128, 256, 256), preserve_range=True)  # preserve_range 会防止将图像归一化
    img_arrflip = img_arr[::-1, :, :]
    # img_arrflip = np.flipud(img_arrflip)

    img_arrback = np.zeros(8388608).reshape((256,256,128))
    # img_arrback = np.zeros(7864320).reshape((256,160,192))
    for i in range(img_arrflip.shape[0]):
        for j in range(img_arrflip.shape[1]):
            for k in range(img_arrflip.shape[2]):
                img_arrback[k, j, i] = img_arrflip[i, j, k]
                if img_arrback[k, j, i] > 400:
                    img_arrback[k, j, i] = 400
                if img_arrback[k, j, i] < -1024:
                    img_arrback[k, j, i] = -1024

    min = np.amin(img_arrback)
    max = np.amax(img_arrback)
    img_arrback = (img_arrback - min) / (max - min)
    img_arrback[img_arrback < 0.0001] = 0
    # img = sitk.GetImageFromArray(img_arrback)
    # sitk.WriteImage(img,r"E:\data\mhacropnew\origincropnewFFD/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")
    np.save(r"E:\data\mhacropnew\origincropnewFFD/case" + str(case) + "/case" + str(case) + "_" + num + ".npy",img_arrback)

def switchtoinhaoSPAREnoinhao(case,num):
    img_flip = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\seg\Creates/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")
    # img_flip = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\seg\COPD/case" + str(case) + "_" + num + ".mha")
    img_arr = sitk.GetArrayFromImage(img_flip)
    img_arr = transform.resize(img_arr, (128, 256, 256), preserve_range=True)  # preserve_range 会防止将图像归一化
    # img_arrflip = img_arr[::-1, :, :]
    # print(img_arr.shape)
    # print(img_arrflip.shape)
    # img_arrflip = np.flipud(img_arrflip)
    # img_arrflip = np.fliplr(img_arrflip)
    img_arrback = np.zeros(8388608).reshape((128,256,256))
    # img_arrback = np.zeros(9175040).reshape((320,224,128))
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            for k in range(img_arr.shape[2]):
                img_arrback[i, j, k] = img_arr[i, j, k]
                if img_arrback[i, j, k] > 400:
                    img_arrback[i, j, k] = 400
                if img_arrback[i, j, k] <= -1000:
                    img_arrback[i, j, k] = -1000

    min = np.amin(img_arrback)
    max = np.amax(img_arrback)
    img_arrback = (img_arrback - min) / (max - min)
    img_arrback[img_arrback < 0.0001] = 0
    # print(img_arrback.dtype)
    # print(img_arrback.max(),img_arrback.min())
    # img = sitk.GetImageFromArray(img_arrback)
    # sitk.WriteImage(img,r"E:\data\merge\seglungEnhance\npy/COPD/case" + str(case) + "_" + num +".mha")
    # sitk.WriteImage(img,r"D:\4DCT\Data\merge\cropwhole\Enhancenpy\Spare/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")

    # np.save(r"E:\data\SPARE\SpareOrigin01/case" + str(case) + "/case" + str(case) + "_" + num + ".npy",img_arrback)
    # np.save(r"E:\data\SPARE\SpareCAD01/val" + "/case" + str(case) + "_" + num + ".npy",img_arrback)
    np.save(r"D:\4DCT\Data\merge\cropwhole\seg\segnpynoinhao/Creates/case" + str(case) + "/case" + str(case) + "_" + num + ".npy",img_arrback)
    # np.save(r"D:\4DCT\Data\merge\cropwhole\seg\segnpynoinhao/COPD/case" + str(case) + "_" + num + ".npy",img_arrback)
def clamp(case,num):
    img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\nii\Spare/case" + str(case) + "/case" + str(case) + "_" + num + "close.nii")
    # img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\nii\Dirlab/case" + str(case) + "_" + num + "close.nii")
    img_arr = sitk.GetArrayFromImage(img)
    img_arr[img_arr>-200] = -200
    img_arr[img_arr<-1000] = -1000
    # print(img_arr.dtype)
    # img_arrback = np.zeros(9175040).reshape((320,224,128))
    # for i in range(img_arr.shape[0]):
    #     for j in range(img_arr.shape[1]):
    #         for k in range(img_arr.shape[2]):
    #             if img_arr[i, j, k] > -200:
    #                 img_arr[i, j, k] = -200
    #             if img_arr[i, j, k] <= -1000:
    #                 img_arr[i, j, k] = -1000
    img = sitk.GetImageFromArray(img_arr)
    sitk.WriteImage(img,r"D:\4DCT\Data\merge\cropwhole\segclamp\Spare/case" + str(case) + "/case" + str(case) + "_" + num + "closeclampe.nii")


    # sitk.WriteImage(img,r"D:\4DCT\Data\merge\Enhance615\-200-1000\COPD/case" + str(case) + "_" + num + ".mha")
def seglung(case, num):
    # mask = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mask\COPD/case" + str(case) + "_" + num + "Mask.mha")
    mask = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mask\Spare/case" + str(case) + "/case" + str(case) + "_" + num + "Mask.mha")
    mask_arr = sitk.GetArrayFromImage(mask)
    mask_arr = mask_arr[::-1, :, :]
    mask_arr = np.flipud(mask_arr)
    # img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\vessel12_0.5\COPD/case" + str(case) + "_" + num + "Enhanceclose.mha")
    img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\vessel12_0.5\Spare/case" + str(case) + "/case" + str(case) + "_" + num + "Enhanceclose.mha")
    img_arr = sitk.GetArrayFromImage(img)
    for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for k in range(img_arr.shape[2]):
                    if mask_arr[i, j, k] == 0:
                        img_arr[i, j, k] = -1000
                    # if mask_arr[i, j, k] != -1024 and img_arr[i, j, k] > 400: #是肺
                    #     img_arr[i, j, k] = 400
    image = sitk.GetImageFromArray(img_arr)

    # sitk.WriteImage(image, r"D:\4DCT\Data\merge\cropwhole\segEnhanceclose\COPD/case" + str(case) + "_" + num + ".mha")
    sitk.WriteImage(image, r"D:\4DCT\Data\merge\cropwhole\segEnhanceclose\Spare/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")

def patch(case,num):
    arr = np.load(r"E:\data\augment320224192\npyclose320224192enhancenew/case" + str(case) + "/case" + str(case) + "_" + num + ".npy")
    for i in range(0,2):
        for j in range(0,2):
            for k in range(0,2):
                patch_arr = arr[i*160:i*160+160, j*112:j*112+112, k*96:k*96+96]
                # patch = sitk.GetImageFromArray(patch_arr)
                # sitk.WriteImage(patch,r"E:\data\newenhancepatch16011296/"+ str(i) + str(j) + str(k) + "/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")
                np.save(r"E:\data\newenhancepatch16011296/"+ str(i) + str(j) + str(k) + "/case" + str(case) + "/case" + str(case) + "_" + num + ".npy",patch_arr)
def resize(case,num):

    arr = np.load(r"E:\data\mhalungairenhance\npy320224192/case" + str(case) + "/case" + str(case) + "_" + num + ".npy")
    # img_flip = sitk.ReadImage("E:\data\mhalungairenhance/" + case + "/" + case + "_" + num + "fusion.mha")
    # img_flip = sitk.ReadImage("E:\data\mhacrop/" + case + "/" + case + "_" + num + ".mha")
    # img_arr = sitk.GetArrayFromImage(img_flip)

    img_arr = transform.resize(arr, (160, 112, 96), preserve_range = True)  # preserve_range 会防止将图像归一化

    min = np.amin(img_arr)
    max = np.amax(img_arr)
    img_arrback = (img_arr - min) / (max - min)
    img_arrback[img_arrback < 0.0001] = 0
    # img = sitk.GetImageFromArray(img_arrback)
    # sitk.WriteImage(img,r"E:\data\mhalungairenhance\npy16011296/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")
    np.save(r"E:\data\mhalungairenhance\npy16011296/case" + str(case) + "/case" + str(case) + "_" + num + ".npy",img_arrback)
    # np.save("E:\data/mhalung256160192/enhance/" + case + "/" + case + "_" + num + ".npy",img_arrback)
def rename(case,num):

    arr = np.load("E:\data\\mhaFFD/case" + str(case) + "/case" + str(case) + "_" + num + ".npy")
    np.save("E:\data/mhaFFD/case" + str(case+10) + "/case" + str(case+10) + "_" + num + ".npy",arr)
def histo(case,num):
    from PIL import Image
    import matplotlib.pyplot as plt
    img_flip = sitk.ReadImage("E:\data\\mhalungenhance336256112/"+ case + "/" + case + "_" + num + ".mha")
    imgarr = sitk.GetArrayFromImage(img_flip)
    arr = imgarr.flatten()
    n, bins, patches = plt.hist(arr,bins=100)
    plt.show()
def ShowShape(case,num):
    img = sitk.ReadImage("E:\data\mhalungairenhance/" + case + "/" + case + "_" + num + "fusion.mha")
    print(img.GetSize())
def add_noise(case,num):
    '''
    添加高斯噪声
    mean : 均值
    var : 方差
    '''
    # image = sitk.ReadImage("E:\data\mhalungairenhance/" + case + "/" + case + "_" + num + "seglungonly.mha")
    # image_arr = sitk.GetArrayFromImage(image)
    image_arr = np.load("E:\data\mhalungairenhance/npyorigin320224192/case" + str(case) + "/case" + str(case) + "_" + num + ".npy")
    # image_arr = image_arr/255
    noise = np.random.normal(0, 0.01, image_arr.shape)
    #产生高斯噪声
    # print(image_arr.shape)
    for i in range(image_arr.shape[0]):
        for j in range(image_arr.shape[1]):
            for k in range(image_arr.shape[2]):
                if image_arr[i,j,k] != 0:
                    image_arr[i,j,k] = image_arr[i,j,k] + noise[i,j,k]
                if image_arr[i,j,k]>1:
                    image_arr[i,j,k] = 1
                if image_arr[i, j, k] < 0:
                    image_arr[i, j, k] = 0

    # image_arr = image_arr*255
    # print(image_arr.max(),image_arr.min())
    # image = sitk.GetImageFromArray(image_arr)
    # sitk.WriteImage(image,"E:\data\mhalungairenhance/npynoise320224192/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")
    np.save("E:\data\mhalungairenhance/npynoise320224192/case" + str(case) + "/case" + str(case) + "_" + num + ".npy",image_arr)
def flip(case,num):
    arr = np.load(r"E:\data\augment320224192\npyclose320224192enhancenew/case" + str(case) + "/case" + str(case) + "_" + num + ".npy")
    arr_flip = np.fliplr(arr)
    np.save(r"E:\data\augment320224192\npyclose320224192enhancenewflip/case" + str(case) + "/case" + str(case) + "_" + num + ".npy",arr_flip)
def FFDflip(case,num):
    img = sitk.ReadImage(r"D:\4DCT\Data\merge\Enhance615_400_-200-1000_01FFD\FFD5\Spare/case" + str(case) + "_" + num + ".nii")
    arr = sitk.GetArrayFromImage(img)
    arr_flip = arr[::-1,:,:]
    arr_flip = np.flipud(arr_flip)
    arr_flip[arr_flip>1]=1
    arr_flip[arr_flip<0]=0
    img = sitk.GetImageFromArray(arr_flip)
    sitk.WriteImage(img,r"D:\4DCT\Data\merge\Enhance615_400_-200-1000_01FFD\FFD5\Spare/case" + str(case) + "_" + num + ".mha")
    # np.save(r"E:\data\augment320224192\npyclose320224192enhancedeform\case" + str(case) + "/case" +str(case) + "_" + num + ".npy",arr_flip)
def resampleVolume(outspacing, vol):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]
    inputspacing = 0
    inputsize = 0
    inputorigin = [0, 0, 0]
    inputdir = [0, 0, 0]

    # 读取文件的size和spacing信息

    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()

    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.5)
    outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.5)
    outsize[2] = int(inputsize[2] * inputspacing[2] / outspacing[2] + 0.5)

    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkBSplineResamplerOrder4)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol
def resample(case,num):
    vol = sitk.Image(sitk.ReadImage(r"E:\data\mha\case" + str(case) + "\case" + str(case) + "_" + num + ".mha"))
    newvol = resampleVolume([1, 1, 2.5], vol)
    wriiter = sitk.ImageFileWriter()
    wriiter.SetFileName(r"E:\data\mha\case" + str(case) + "\case" + str(case) + "_" + num + "resample.mha")
    wriiter.Execute(newvol)
def checkSpacing(case):
    # img = sitk.ReadImage(r"E:\data\mha/case" + str(case) + "/case" + str(case) + "_T00.mha")
    img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mha\Dirlab/case" + str(case) + "_T00.mha")
    print(img.GetSize())
def CurvatureAnisotropicDiffusionImageFilter(case,num):
    # arr = np.load(r"D:\4DCT\Data\merge\cropwhole\seg\segnpy\Spare\case" + str(case) + "\case" + str(case) + "_" + num + ".npy")
    arr = np.load(r"D:\4DCT\Data\merge\cropwhole\seg\segnpy\NLST\case" + str(case) + "_" + num + ".npy")
    # img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mha\Spare\case" + str(case) + "\case" + str(case) + "_" + num + ".mha",sitk.sitkFloat32)
    # img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mha\Dirlab\case" + str(case) + "_" + num + ".mha",sitk.sitkFloat32)
    img = sitk.GetImageFromArray(arr)
    sitk_src_gaus = sitk.CurvatureAnisotropicDiffusionImageFilter()
    sitk_src_gaus.SetTimeStep(0.08)
    # sitk_src_gaus.SetMaximumError(0.1)
    sitk_src_gaus = sitk_src_gaus.Execute(img)
    # sitk.WriteImage(sitk_src_gaus, r"D:\4DCT\Data\merge\cropwhole\CADmha\Spare\case" + str(case) + "\case" + str(case) + "_" + num + ".mha")
    # sitk.WriteImage(sitk_src_gaus, r"D:\4DCT\Data\merge\cropwhole\CADmha\Dirlab\case" + str(case) + "_" + num + ".mha")
    arr = sitk.GetArrayFromImage(sitk_src_gaus)
    # np.save(r"D:\4DCT\Data\merge\cropwhole\segCAD\segCADnpy\Spare/case" + str(case) + "\case" + str(case) + "_" + num + ".npy",arr)
    np.save(r"D:\4DCT\Data\merge\cropwhole\segCAD\segCADnpy\NLST/case" + str(case) + "_" + num + ".npy",arr)
def Guassiansmo(case,num):
    # arr = np.load(r"D:\4DCT\Data\merge\cropwhole\npy\Creates\case" + str(case) + "\case" + str(case) + "_" + num + ".npy")
    arr = np.load(r"D:\4DCT\Data\merge\cropwhole\seg/segnpy\NLST\case" + str(case) + "_" + num + ".npy")
    # img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mha\Spare\case" + str(case) + "\case" + str(case) + "_" + num + ".mha",sitk.sitkFloat32)
    # img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mha\Dirlab\case" + str(case) + "_" + num + ".mha",sitk.sitkFloat32)
    img = sitk.GetImageFromArray(arr)
    sitk_src_gaus = sitk.DiscreteGaussianImageFilter()
    sitk_src_gaus.SetVariance(5)
    # sitk_src_gaus.SetTimeStep(0.08)
    # sitk_src_gaus.SetMaximumError(0.1)
    sitk_src_gaus = sitk_src_gaus.Execute(img)
    # sitk.WriteImage(sitk_src_gaus, r"D:\4DCT\Data\merge\cropwhole\CADmha\Spare\case" + str(case) + "\case" + str(case) + "_" + num + ".mha")
    # sitk.WriteImage(sitk_src_gaus, r"D:\4DCT\Data\merge\cropwhole\CADmha\Dirlab\case" + str(case) + "_" + num + ".mha")
    arr = sitk.GetArrayFromImage(sitk_src_gaus)
    # np.save(r"D:\4DCT\Data\merge\cropwhole\Gaussiannpy\Creates/case" + str(case) + "\case" + str(case) + "_" + num + ".npy",arr)
    np.save(r"D:\4DCT\Data\merge\cropwhole\Gaussiannpy\NLST/case" + str(case) + "_" + num + ".npy",arr)

def fusion(case,num):
    vessel = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\segclamp/Spare/case" + str(case) + "/case" + str(case) + "_" + num + "vesselclamp.nii")
    # vessel = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\segclamp\Dirlab/case" + str(case) + "_" + num + "vesselclamp.nii")
    # vessel = sitk.ReadImage(r"D:\4DCT\Data\COPD\crop/" + case + "_" + num + "vessel.nii.gz")

    vessel_arr = sitk.GetArrayFromImage(vessel)
    vessel_arr = vessel_arr[:,::-1,::-1]
    # vessel_arr = np.fliplr(vessel_arr)
    # vessel_arr = np.flipud(vessel_arr)
    # print("vessel_arr ", vessel_arr.shape)
    # img = sitk.ReadImage(r"D:\4DCT\Data\COPD\crop/" + case + "_" + num + "lungseg.nii.gz")
    # img = sitk.ReadImage(r"E:\data\mhacropnew\case" + str(case) + "/case" + str(case) + "_" + num + "close.nii")
    # mask = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mask\NLST/case" + str(case) + "_" + num + "Mask.mha")
    # # mask = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mask\Spare/case" + str(case) + "/case" + str(case) + "_" + num + "Mask.mha")
    # mask_arr = sitk.GetArrayFromImage(mask)
    # mask_arr = mask_arr[::-1, :, :]
    # mask_arr = np.flipud(mask_arr)
    # img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\seg\Dirlab/case" + str(case) + "_" + num + ".mha")
    img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\seg\Spare/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")
    img_arr = sitk.GetArrayFromImage(img)
    # print("img_arr ", img_arr.shape)
    for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for k in range(img_arr.shape[2]):
                    if (vessel_arr[i, j, k] != 0) and (img_arr[i, j, k] != -1000):
                        img_arr[i, j, k] = img_arr[i, j, k] + 300 * (vessel_arr[i, j, k])
                    # if (vessel_arr[i, j, k] >= 0.5) and (img_arr[i, j, k] != -1000):
                    #     img_arr[i, j, k] = img_arr[i, j, k] + 400*(vessel_arr[i, j, k])
                    # if (vessel_arr[i, j, k] < 0.5) and (img_arr[i, j, k] != -1000):
                    #     img_arr[i, j, k] = img_arr[i, j, k] + 800*(vessel_arr[i, j, k])


    fusion = sitk.GetImageFromArray(img_arr)
    # sitk.WriteImage(fusion,r"D:\4DCT\Data\merge\cropwhole\vessel712\Dirlab/case" + str(case) + "_" + num + ".mha")
    sitk.WriteImage(fusion,r"D:\4DCT\Data\merge\cropwhole\vessel712\Spare/case" + str(case)  + "/case" + str(case) + "_" + num + ".mha")

def vesselcrop(case,num):
    # vessel = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\vesselanother\Dirlab/case" + str(case) + "_" + num + "vessel2.nii")
    vessel = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\vesselanother\Spare/case" + str(case) + "/case" + str(case) + "_" + num + "vessel2.nii")
    # vessel = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\segclamp\Dirlab/case" + str(case) + "_" + num + "vesselclamp.nii")
    # vessel = sitk.ReadImage(r"D:\4DCT\Data\COPD\crop/" + case + "_" + num + "vessel.nii.gz")
    vessel.SetOrigin((0, 0, 0))
    vessel_arr = sitk.GetArrayFromImage(vessel)
    # vessel_arr = (vessel_arr - vessel_arr.min()) / (vessel_arr.max() - vessel_arr.min())
    vessel_arr = vessel_arr[:, ::-1, ::-1]

    img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\seg\Spare/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")
    # img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\seg\Dirlab/case" + str(case) + "_" + num + ".mha")
    img_arr = sitk.GetArrayFromImage(img)
    # print("img_arr ", img_arr.shape)
    for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for k in range(img_arr.shape[2]):
                    if (img_arr[i, j, k] == -1000):
                        vessel_arr[i, j, k] = 0


    vessel = sitk.GetImageFromArray(vessel_arr)

    # sitk.WriteImage(vessel,r"D:\4DCT\Data\merge\cropwhole\vesselanother\Dirlab/case" + str(case) + "_" + num + "vessel2crop.nii")
    sitk.WriteImage(vessel,r"D:\4DCT\Data\merge\cropwhole\vesselanother\Spare/case" + str(case) + "/case" + str(case) + "_" + num + "vessel2crop.nii")

def switchtoinhaoSPARE(case,num):
    img_flip = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\vessel712\Spare/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")
    # img_flip = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\vessel712\Dirlab/case" + str(case) + "_" + num + ".mha")
    img_arr = sitk.GetArrayFromImage(img_flip)
    img_arr = transform.resize(img_arr, (128, 256, 256), preserve_range=True)  # preserve_range 会防止将图像归一化
    img_arrflip = img_arr[::-1, :, :]
    # img_arrflip = np.flipud(img_arrflip)
    # img_arrflip = np.fliplr(img_arrflip)

    img_arrback = np.zeros(8388608).reshape((256,256,128))
    # img_arrback = np.zeros(12582912).reshape((256,256,192))
    for i in range(img_arrflip.shape[0]):
        for j in range(img_arrflip.shape[1]):
            for k in range(img_arrflip.shape[2]):
                img_arrback[k, j, i] = img_arrflip[i, j, k]

    img_arrback[img_arrback > 400] = 400
    img_arrback[img_arrback < -1000] = -1000
                # if img_arrback[k, j, i] > 400:
                #     img_arrback[k, j, i] = 400
                # if img_arrback[k, j, i] <= -1000:
                #     img_arrback[k, j, i] = -1000

    min = np.amin(img_arrback)
    max = np.amax(img_arrback)
    img_arrback = (img_arrback - min) / (max - min)
    img_arrback[img_arrback < 0.0001] = 0
    # print(img_arrback.dtype)
    # print(img_arrback.max(),img_arrback.min())
    # img = sitk.GetImageFromArray(img_arrback)
    # sitk.WriteImage(img,r"D:\4DCT\Data\merge\cropwhole\seg\segnpy320224112/Dirlab/case" + str(case) + "_" + num +"re0.5.mha")
    # sitk.WriteImage(img,r"D:\4DCT\Data\merge\cropwhole\vesseltest2\vesseltest2npy/Dirlab/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")

    # np.save(r"E:\data\SPARE\SpareOrigin01/case" + str(case) + "/case" + str(case) + "_" + num + ".npy",img_arrback)
    # np.save(r"E:\data\SPARE\SpareCAD01/val" + "/case" + str(case) + "_" + num + ".npy",img_arrback)
    np.save(r"D:\4DCT\Data\merge\cropwhole\vessel712\vessel712npy/Spare/case" + str(case) + "/case" + str(case) + "_" + num + ".npy",img_arrback)
    # np.save(r"D:\4DCT\Data\merge\cropwhole\vessel712\vessel712npy\Dirlab/case" + str(case) + "_" + num + ".npy",img_arrback)

def fusion2(case,num):
    # vessel = sitk.ReadImage(r"C:\Users\Wuweb\Desktop\4DCTpaper\data\vesselcase8_T00.nii")
    # vessel = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\segclamp\Dirlab/case" + str(case) + "_" + num + "vesselclamp.nii")
    # vessel = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\vesselanother\COPD/case" + str(case) + "_" + num + "vessel2crop.nii")
    vessel = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\vesselanother\Spare/case" + str(case) + "/case" + str(case) + "_" + num + "vessel2.nii")
    vessel.SetOrigin((0,0,0))
    vessel_arr = sitk.GetArrayFromImage(vessel)
    vessel_arr = (vessel_arr - vessel_arr.min()) / (vessel_arr.max() - vessel_arr.min())
    # vessel_arr = vessel_arr[:, ::-1, ::-1]
    # vessel = sitk.GetImageFromArray(vessel_arr)
    # sitk.WriteImage(vessel,"vessel.mha")
    # os.system("pause")


    # vessel_arr = np.fliplr(vessel_arr)
    # vessel_arr = np.flipud(vessel_arr)
    # print("vessel_arr ", vessel_arr.shape)
    # img = sitk.ReadImage(r"D:\4DCT\Data\COPD\crop/" + case + "_" + num + "lungseg.nii.gz")
    # img = sitk.ReadImage(r"E:\data\mhacropnew\case" + str(case) + "/case" + str(case) + "_" + num + "close.nii")
    # mask = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mask\NLST/case" + str(case) + "_" + num + "Mask.mha")
    # # mask = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\mask\Spare/case" + str(case) + "/case" + str(case) + "_" + num + "Mask.mha")
    # mask_arr = sitk.GetArrayFromImage(mask)
    # mask_arr = mask_arr[::-1, :, :]
    # mask_arr = np.flipud(mask_arr)
    # img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\seg\COPD/case" + str(case) + "_" + num + ".mha")
    img = sitk.ReadImage(r"D:\4DCT\Data\merge\cropwhole\seg\Spare/case" + str(case) + "/case" + str(case) + "_" + num + ".mha")
    img_arr = sitk.GetArrayFromImage(img)
    # print("img_arr ", img_arr.shape)
    for i in range(img_arr.shape[0]):
            for j in range(img_arr.shape[1]):
                for k in range(img_arr.shape[2]):
                    if (vessel_arr[i, j, k] != 0) and (img_arr[i, j, k] != -1000):
                        img_arr[i, j, k] = img_arr[i, j, k] + 2000 * (vessel_arr[i, j, k])
                    # if (vessel_arr[i, j, k] >= 0.5) and (img_arr[i, j, k] != -1000):
                    #     img_arr[i, j, k] = img_arr[i, j, k] + 400*(vessel_arr[i, j, k])
                    # if (vessel_arr[i, j, k] < 0.5) and (img_arr[i, j, k] != -1000):
                    #     img_arr[i, j, k] = img_arr[i, j, k] + 800*(vessel_arr[i, j, k])


    fusion = sitk.GetImageFromArray(img_arr)
    # sitk.WriteImage(fusion,r"D:\4DCT\Data\merge\cropwhole\vessel712\Dirlab/case" + str(case) + "_" + num + ".mha")
    # sitk.WriteImage(fusion,r"D:\4DCT\Data\merge\cropwhole\vesselanother\COPD/case" + str(case) + "_" + num + "vessel2ENH.mha")
    sitk.WriteImage(fusion,r"D:\4DCT\Data\merge\cropwhole\vesselanother\Spare/case" + str(case) + "/case" + str(case) + "_" + num + "vessel2ENH.mha")

def iterCurvatureAnisotropicDiffusionImageFilter(case,num):
    arr = np.load(r"D:\4DCT\Data\merge\cropwhole\seg\segnpy\Spare\case" + str(case) + "\case" + str(case) + "_" + num + ".npy")
    # arr = np.load(r"D:\4DCT\Data\merge\cropwhole\seg\segnpy\COPD\case" + str(case) + "_" + num + ".npy")
    img = sitk.GetImageFromArray(arr)

    sitk_src_gaus = sitk.CurvatureAnisotropicDiffusionImageFilter()
    sitk_src_gaus.SetTimeStep(0.08)
    img = sitk_src_gaus.Execute(img)
    img = sitk_src_gaus.Execute(img)
    img = sitk_src_gaus.Execute(img)
    img = sitk_src_gaus.Execute(img)
    img = sitk_src_gaus.Execute(img)
    arr = sitk.GetArrayFromImage(img)
    np.save(r"D:\4DCT\Data\merge\cropwhole\segCAD5\segCAD5npy\Spare/case" + str(case) + "\case" + str(case) + "_" + num + ".npy",arr)
    # np.save(r"D:\4DCT\Data\merge\cropwhole\segCAD5\segCAD5npy\COPD/case" + str(case) + "_" + num + ".npy", arr)

def stochastic_resonance_denoising(case,num):
    noise_strength = 0.005  # 噪声强度
    threshold = 0.001  # 阈值
    # image_array = np.load(r"D:\4DCT\Data\traincascade\segnpy\Creates\case" + str(case) + "\case" + str(case) + "_" + num + ".npy")
    image_array = np.load(r"D:\4DCT\Data\traincascade\segnpy\Dirlab\case" + str(case) + "_" + num + ".npy")
    noisy_image_array = image_array + np.random.normal(0, noise_strength, image_array.shape)
    denoised_image_array = np.where(np.abs(noisy_image_array) > threshold, noisy_image_array, 0)
    # np.save(r"D:\4DCT\Data\traincascade\segnpySR\Creates/case" + str(case) + "\case" + str(case) + "_" + num + ".npy", denoised_image_array)
    np.save(r"D:\4DCT\Data\traincascade\segnpySR\Dirlab/case" + str(case) + "_" + num + ".npy", denoised_image_array)

def stochastic_resonance_denoising2(case,num):

    threshold = 0.01  # 阈值
    # ct_array = np.load(r"D:\4DCT\Data\traincascade\segnpy\Dirlab\case" + str(case) + "_" + num + ".npy")
    ct_array = np.load(r"D:\4DCT\Data\traincascade\segnpy\Creates/case" + str(case) + "\case" + str(case) + "_" + num + ".npy")

    # 对每个切片进行小波分解和重构
    denoised_ct_image = np.zeros_like(ct_array)

    for i in range(ct_array.shape[1]):  # 修改这里的索引以展示横断位视图
        # 小波分解
        coeffs = pywt.wavedec2(ct_array[:, i, :], wavelet='db1', level=4)
        # 阈值处理
        coeffs_thresholded = []
        for coeff in coeffs:
            if isinstance(coeff, tuple):
                # 处理多层小波分解的结果
                coeff_slices = []
                for c in coeff:
                    coeff_slices.append(pywt.threshold(c, threshold * np.max(c), mode='soft'))
                coeffs_thresholded.append(tuple(coeff_slices))
            else:
                # 处理单层小波分解的结果
                coeffs_thresholded.append(pywt.threshold(coeff, threshold * np.max(coeff), mode='soft'))
        # 小波逆变换
        denoised_slice = pywt.waverec2(coeffs_thresholded, wavelet='db1')
        # 调整处理后的切片大小，保持与原始切片一致
        denoised_slice_resized = cv2.resize(denoised_slice, (ct_array.shape[2], ct_array.shape[0]))
        # 保存处理后的切片
        denoised_ct_image[:, i, :] = denoised_slice_resized
    # 将处理后的切片中的NaN值改为0
    denoised_ct_image = np.nan_to_num(denoised_ct_image, nan=0)
    # np.save(r"D:\4DCT\Data\traincascade\segnpySR2\Dirlab/case" + str(case) + "_" + num + ".npy", denoised_ct_image)
    np.save(r"D:\4DCT\Data\traincascade\segnpySR2\Creates/case" + str(case) + "\case" + str(case) + "_" + num + ".npy", denoised_ct_image)

    # output_image = sitk.GetImageFromArray(denoised_ct_image)
    # output_file_path = 'SR.mha'
    # sitk.WriteImage(output_image, output_file_path)

def Preprogress():
        for case in cases:
            for num in nums:
                # stochastic_resonance_denoising(case, num)
                stochastic_resonance_denoising2(case, num)
                # mhatonpy(case,num)
                # niitomha(case,num)
                # npytomha(case,num)
                # npytonii(case, num)
                # Close(case,num)
                # mhatonii(case, num)
                # fusion2(case,num)
                # vesselcrop(case, num)
                # switchtoinhaoSPARE(case,num)
                # switchtoinhaoSPAREnoinhao(case, num)
                # seglung(case, num)
                # clamp(case,num)
                # switchtoinhao(case,num)
                # Guassiansmo(case,num)
                # CurvatureAnisotropicDiffusionImageFilter(case,num)
                # iterCurvatureAnisotropicDiffusionImageFilter(case, num)
                # switchtomaomao(case, num)
                # switchtoinhaoNoresize(case, num)
                # histo(case,num)
                # mhatonpy(case, num)

                # resample(case,num)
                # ShowShape(case,num)
                # resize(case,num)
                # rename(case,num)
                # add_noise(case, num)
                # Rename(case,num)
                # flip(case,num)
                # FFDflip(case,num)
                # patch(case,num)
                # resample(case, num)
                # checkSpacing(case)
                print(case, num)


# cases = ['case1','case2','case3','case4','case5','case6','case7','case8','case9','case10','case11','case12','case13','case14','case15','case16']
# cases = [1]
# cases = [1,2,3,4,5,6,7,8,9,10]
cases = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
# cases = [19,20,21,22]
# cases = [11,12,13,14,15,16]
# cases = [8]
# cases = ['case1','case2','case3','case4','case5','case6','case7','case8','case9','case10']
nums = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90']
# nums = ["T00","T50"]
# nums = ['T00']
# cases = ['case1']

if __name__ == '__main__':

    Preprogress()
