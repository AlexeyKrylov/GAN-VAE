import numpy as np
import cv2
import matplotlib.pyplot as plt
import rle_coding
import pandas as pd
import skimage.io as tifffile
from sklearn.cluster import KMeans
from scipy import ndimage


def median_cut_quantize(img, img_arr):
    # when it reaches the end, color quantize
    # print("to quantize: ", len(img_arr))
    r_average = np.mean(img_arr[:, 0])
    g_average = np.mean(img_arr[:, 1])
    b_average = np.mean(img_arr[:, 2])

    for data in img_arr:
        img[data[3]][data[4]] = [r_average, g_average, b_average]


def split_into_buckets(img, img_arr, depth):
    if len(img_arr) == 0:
        return

    if depth == 0:
        median_cut_quantize(img, img_arr)
        return

    r_range = np.max(img_arr[:, 0]) - np.min(img_arr[:, 0])
    g_range = np.max(img_arr[:, 1]) - np.min(img_arr[:, 1])
    b_range = np.max(img_arr[:, 2]) - np.min(img_arr[:, 2])

    space_with_highest_range = 0

    if g_range >= r_range and g_range >= b_range:
        space_with_highest_range = 1
    elif b_range >= r_range and b_range >= g_range:
        space_with_highest_range = 2
    elif r_range >= b_range and r_range >= g_range:
        space_with_highest_range = 0

    # print("space_with_highest_range:", space_with_highest_range)

    # sort the image pixels by color space with highest range
    # and find the median and divide the array.
    img_arr = img_arr[img_arr[:, space_with_highest_range].argsort()]
    median_index = int((len(img_arr) + 1) / 2)
    # print("median_index:", median_index)

    # split the array into two buckets along the median
    split_into_buckets(img, img_arr[0:median_index], depth - 1)
    split_into_buckets(img, img_arr[median_index:], depth - 1)



def PoissonBlending(source, target, mask, out, center):
    F = cv2.seamlessClone(src=source, dst=target, mask=mask, p=center, flags=cv2.NORMAL_CLONE)
    cv2.imwrite(out, F)

def KMeansQC(S, N_CLUSTERS):
    (h, w) = S.shape[:2]
    S_ = cv2.cvtColor(S, cv2.COLOR_BGR2LAB)
    S_ = S_.reshape((S_.shape[0] * S_.shape[1], 3))
    clt = KMeans(n_clusters=N_CLUSTERS)
    labels = clt.fit_predict(S_)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    return quant

def median_cutQC(S, n_colors):
    pow = np.log2(n_colors).astype(np.int64)
    flattened_img_array = []
    for rindex, rows in enumerate(S):
        for cindex, color in enumerate(rows):
            flattened_img_array.append([color[0], color[1], color[2], rindex, cindex])
    flattened_img_array = np.array(flattened_img_array)
    split_into_buckets(S, flattened_img_array, pow)
    return S

def MPB(source, target, mask, out, center):
    F = cv2.seamlessClone(src=source, dst=target, mask=mask, p=center, flags=cv2.NORMAL_CLONE)
    F = cv2.seamlessClone(src=target, dst=F, mask=255 - mask, p=(F.shape[1] // 2, F.shape[0] // 2), flags=cv2.NORMAL_CLONE)
    cv2.imwrite(out, F)

def MPB2(source, target, mask, center, name, debug=False):
    mask[mask == 85] = 230
    mask[mask == 170] = 100
    T = 64
    N_COLORS = 4
    K_BLUR = 20
    K_GBLUR = 99
    sigma = 0.1
    out = f"imgs/results/result_{name}.jpg"
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(2, 4, 3)
        plt.imshow(target)
        ax.set_title("target")
        ax = fig.add_subplot(2, 4, 4)
        plt.imshow(source)
        ax.set_title("source")
        ax = fig.add_subplot(2, 4, 1)

    """
    First step:

        The source image is used as a target image in the
        original Poisson blending operation and the target
        image is used as a source image. The original binary
        mask is inverted, so the blended region is the inverse
        of the specified area in the original binary mask. The
        first step creates a composited image I1.

    """
    source1 = target
    target1 = (target * 0.999).astype(np.uint8)
    target1[center[0] - source.shape[0] // 2: center[0] + source.shape[0] // 2,
    center[1] - source.shape[1] // 2: center[1] + source.shape[1] // 2, :] = source
    mask1 = np.full(target.shape, 255, dtype=np.uint8)
    mask1[center[0] - source.shape[0] // 2: center[0] + source.shape[0] // 2,
    center[1] - source.shape[1] // 2: center[1] + source.shape[1] // 2, :] = 255 - mask
    out1 = f"imgs/tmp/temp1_{name}.jpg"
    MPB(source=source1, target=target1, mask=mask1, out=out1, center=(target1.shape[1] // 2, target1.shape[0] // 2))

    """
    Second step:
        I1 which is used as a source image in the second step of the MPB. 
        We use the original target image as a target image in the
        Poisson blending operation and we use the original
        binary mask, to get the second composited image I2.
    """
    I1 = cv2.imread(f"imgs/tmp/temp1_{name}.jpg")
    if debug:
        plt.imshow(I1)
        ax.set_title("I1")
    source2 = I1
    target2 = target
    mask2 = 255 - mask1
    out2 = f"imgs/tmp/temp2_{name}.jpg"
    MPB(source=source2, target=target2, mask=mask2, out=out2, center=center)

    """
    Third step:
        The generated images from the previous steps are blended using an alpha
        blending operation. We use the original target image
        and the generated image from the second step to determine the binary mask. 
    """
    I2 = cv2.imread(f"imgs/tmp/temp2_{name}.jpg")
    if debug:
        ax = fig.add_subplot(2, 4, 5)
        plt.imshow(I2)
        ax.set_title("I2")
    S = np.abs(I2 - target)
    S_ = median_cutQC(S, N_COLORS)
    S_[S_ <= T] = 0
    S_[S_ > 0] = 1
    M = np.float32(ndimage.binary_fill_holes(S_))
    M = cv2.blur(M, (K_BLUR, K_BLUR))
    M = cv2.GaussianBlur(src=M, ksize=(K_GBLUR, K_GBLUR), sigmaX=sigma, dst=M, sigmaY=sigma)
    I1 = I1.astype(float)
    I2 = I2.astype(float)
    alpha = M.astype(float)
    if debug:
        ax = fig.add_subplot(2, 4, 7)
        plt.imshow(alpha)
        ax.set_title("alpha")
    I1 = cv2.multiply(alpha, I1).astype("uint8")
    if debug:
        ax = fig.add_subplot(2, 4, 2)
        plt.imshow(I1)
        ax.set_title("I1 * alpha")
    I2 = cv2.multiply(1.0 - alpha, I2).astype("uint8")
    if debug:
        ax = fig.add_subplot(2, 4, 6)
        plt.imshow(I2)
        ax.set_title("I2 * (1 - alpha)")
    outImage = cv2.add(I1, I2)
    if debug:
        ax = fig.add_subplot(2, 4, 8)
        plt.imshow(outImage)
        ax.set_title("Result")
        plt.savefig(f"imgs/results/debug1_{name}.jpg")
        #plt.show()
    cv2.imwrite(out, outImage)

def load_and_preprocess(target_file, annotations_file, index, debug=False):
    masks = pd.read_csv(annotations_file)
    name = target_file[16:-4] + '_' + str(masks.name[index])[9:-4]
    source = (tifffile.imread("imgs/" + str(masks.name[index])) * 255).astype(np.uint8)
    source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
    if debug:
        print(f"source shape: {source.shape}")
        print(f"source name: {str(masks.name[index])}")
    mask = (rle_coding.rle_decode(masks.drop(columns=["name"]).iloc[index], (128, 128))[1] * 255 / 3).astype(np.uint8)
    mask = np.repeat(mask, 3, axis=1).reshape((128, 128, 3))
    target = cv2.imread(target_file)
    if debug:
        print(f"mask shape: {mask.shape}")
        print(f"target shape: {target.shape}")
    return source, target, mask, name


if __name__ == "__main__":
    backgrounds = ["imgs/background/pict_04_2_3.png",
                   "imgs/background/pict_04_7.png",
                   "imgs/background/pict_09-33-25_1_0.png"]
    _, target1, _, _ = load_and_preprocess(target_file="imgs/background/pict_04_2_3.png", annotations_file="imgs/target_annot.csv", index=0,debug=True)
    for n, b in enumerate(backgrounds):
        for i in range(5):
            source, target, mask, name = load_and_preprocess(target_file=b, annotations_file="imgs/target_annot.csv", index=i, debug=True)
            if n == 0:
                #center = (430, 880)
                target = target[430-250:430+250, 880-250:880+250]
            elif n == 1:
                #center = (1450, 1200)
                target = target[1450 - 250:1450 + 250, 1200 - 100:1200 + 400]
            else:
                #center = (635, 1365)
                target = target[635 - 250:635 + 250, 1365 - 400:1365 + 100]
            MPB2(source=source, target=target, mask=mask, center=(250, 250), name=name, debug=True)

            if i == 0:
                name += "tree"
                source = target1[93:93+38, 130:130+84]
                mask[mask > -1] = 255
                mask = mask[:38, :84]
                MPB2(source=source, target=target, mask=mask, center=(250, 250), name=name, debug=True)