import tensorflow as tf
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from math import log10, sqrt
import argparse
import cv2

def load_graph(frozen_graph_filename):
    """
    Args:
        frozen_graph_filename (str): Full path to the .pb file.
    """
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
        return graph


def segment(graph, image_file):
    """
    Does the segmentation on the given image.
    Args:
        graph (Tensorflow Graph)
        image_file (str): Full path to your image
    Returns:
        segmentation_mask (np.array): The segmentation mask of the image.
    """
    # We access the input and output nodes
    x = graph.get_tensor_by_name('prefix/ImageTensor:0')
    y = graph.get_tensor_by_name('prefix/SemanticPredictions:0')

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # We launch a Session
    with tf.compat.v1.Session(graph=graph) as sess:
        image = Image.open(image_file)
        image = image.resize((250, 125))
        image_array = np.array(image)
        image_array = image_array[:, :, :3]
        image_array = np.expand_dims(image_array, axis=0)

        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        pred = sess.run(y, feed_dict={x: image_array})

        pred = pred.squeeze()

    return pred


def get_n_rgb_colors(n):
    """
    Get n evenly spaced RGB colors.
    Returns:
        rgb_colors (list): List of RGB colors.
    """
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    rgb_colors = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

    return rgb_colors


def parse_pred(pred, n_classes):
    """
    Parses a prediction and returns the prediction as a PIL.Image.
    Args:
        pred (np.array)
    Returns:
        parsed_pred (PIL.Image): Parsed prediction that we can view as an image.
    """
    uni = np.unique(pred)

    empty = np.empty((pred.shape[0], pred.shape[1], 3))

    colors = get_n_rgb_colors(n_classes)

    for i, u in enumerate(uni):
        idx = np.transpose((pred == u).nonzero())
        c = colors[u]
        empty[idx[:, 0], idx[:, 1]] = [c[0], c[1], c[2]]

    parsed_pred = np.array(empty, dtype=np.uint8)
    parsed_pred = Image.fromarray(parsed_pred)

    return parsed_pred


strlabel = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
            5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation',
            9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
            14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'}


def PSNR(org, prt):
    org = np.array(org)
    prt = np.array(prt)

    # removing alpha channel if any
    org = org[:, :, :3]
    prt = prt[:, :, :3]

    mse = np.mean((org - prt) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--IMAGE_FILE', help='input image name', default='berlin_000000_000019_leftImg8bit.png')
    parser.add_argument('--perturb_image', help='perturbed image name', default='rain.png')
    args = parser.parse_args()
    # N_CLASSES = 33
    MODEL_FILE = 'frozen_inference_graph.pb '
    IMAGE_FILE = args.IMAGE_FILE

    # IMAGE_FILE = 'real_scale.png'

    graph = load_graph(MODEL_FILE)
    prediction = segment(graph, IMAGE_FILE)
    from matplotlib import pyplot as plt

    print('before', [strlabel[v] for v in np.unique(prediction)])

    perturb_img = args.perturb_img
    prediction_per = segment(graph, perturb_img)

    print('after', [strlabel[v] for v in np.unique(prediction_per)])

    org_img = Image.open(IMAGE_FILE)
    prt_img = Image.open(perturb_img)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Perturbations affect')
    org_img = org_img.resize((250, 125))
    axs[0, 0].imshow(org_img)
    axs[0, 0].title.set_text('Original Image')

    prt_img = prt_img.resize((250, 125))

    psnr = PSNR(org_img, prt_img)
    print(psnr)

    axs[0, 1].imshow(prt_img)
    axs[0, 1].title.set_text('perturbed image')

    prs = np.array(parse_pred(prediction_per, 18))
    axs[1, 1].imshow(prs)
    axs[1, 1].title.set_text('perturbed segmented')

    axs[1, 0].imshow(np.array(parse_pred(prediction, 18)))
    axs[1, 0].title.set_text('Original segmented ')

    plt.show()


