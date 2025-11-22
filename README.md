# UpscalingImageTransformer

## Data.py
https://www.kaggle.com/datasets/evgeniumakov/images4k

The Dataset provided above is the Images4k dataset, it contains ~2000 images of resolution ~(3840 x 2160). These images are loaded using OpenCV in RGB before a number of actions are performed to prep the data. First a blur is applied to 20% of all images, then images are randomly resized to a smaller image. Next, gaussian noise is applied to 20% of images. Lastly 50% of images are compressed using the JPEG Compression Algorithm. The output is normalized between 0 - 1 and then fed into the model. Getting the target image is similar, however the image is not altered outside of being resized. The alterations are done in order to manage a robust variety of artifacts and alterations that might occur in an image.

## Files & Usage
- config.ini
  - Specify model and datset paths to use
- data.py
  - Loading images
  - Processing images
- esrgan.py
  - Generator class
  - Discriminator class
  - GAN train loop
  - Add `--test` flag for testing (default: training)
- rrdbnet_4x.py
  - Base 4x upscale model
  - Train loop
  - Add `--test` flag for testing (default: training)
- rrdbnet_16x.py
  - Base 16x upscale model
  - Train loop
  - Testing
  - Add `--test` flag for testing (default: training)
- test_model.py
  - Testing models
  - Comparing models
- upscale.py
  - Testing models on specific images

## Example Usage
```shell
python esrgan.py --test
```