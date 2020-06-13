# execution command: python preprocessing.py --input-dir lfw --output-dir lfw_alig --crop-dim 224

from align_dlib import AlignDlib
import argparse
import glob
import logging
import multiprocessing as mp
import os
import time
import cv2

logger = logging.getLogger(__name__)

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

# Detect face, align and crop
def preprocess_image(input_path, output_path, crop_dim):
    image = _process_image(input_path, crop_dim)
    if image is not None:
        logger.debug('Writing processed file: {}'.format(output_path))
        cv2.imwrite(output_path, image)
    else:
        logger.warning("Skipping filename: {}".format(input_path))

# Process of image 
def _process_image(filename, crop_dim):
    image = None
    aligned_image = None
    
    logger.debug('Reading image: {}'.format(filename))
    image = cv2.imread(filename, )
    # Color channels are changed from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is not None:
        aligned_image = _align_image(image, crop_dim)
    else:
        raise IOError('Error buffering image: {}'.format(filename))

    return aligned_image


# This functions aligns faces that are centered 
# based on inner eyes bottom lips
def _align_image(image, crop_dim):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', default='data', dest='input_dir')
    parser.add_argument('--output-dir', type=str, action='store', default='output', dest='output_dir')
    parser.add_argument('--crop-dim', type=int, action='store', default=224, dest='crop_dim',
                        help='Size to crop images to')

    args = parser.parse_args()
    
    # Start a timer in order to calculate the duration of datasets preprocessing
    start_time = time.time()
    pool = mp.Pool(processes=mp.cpu_count())

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for image_dir in os.listdir(args.input_dir):
        image_output_dir = os.path.join(args.output_dir, os.path.basename(os.path.basename(image_dir)))
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

    image_paths = glob.glob(os.path.join(args.input_dir, '**/*.jpg'))
    for index, image_path in enumerate(image_paths):
        image_output_dir = os.path.join(args.output_dir, os.path.basename(os.path.dirname(image_path)))
        output_path = os.path.join(image_output_dir, os.path.basename(image_path))
        pool.apply_async(preprocess_image, (image_path, output_path, args.crop_dim))

    pool.close()
    pool.join()
    logger.info('Completed in {} seconds'.format(time.time() - start_time))