/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/*Here we build a tf.js model to recognize handwritten digits with CNN
*Convolutional Neural Network is a type of artificial neural n/w
*used in image recognition and proccessing ,that is specifically designed to process pixel data*/

//the size of an image(w=28,h=28)28x28=784
const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

//path to the image and labels
/*image sprite is a collection of images put it into a single image
*a webpage with many images can take long time to load
*and generates multiple server requests,image sprites reduces the number of server request.*/

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
 /*export*/ class MnistData {
    constructor() {
      this.shuffledTrainIndex = 0;
      this.shuffledTestIndex = 0;
    }

//load() responsible for asynchronously loading the image and label data
  async load() {
// Make a request for the MNIST sprited image.

    const img = new Image();

    //canvas() is a DOM element that provides easy access to pixel arrays
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

/*getContext method returns an object that provides methods and properties for drawing on canvas
*getContext(2d) is used to draw lines,boxes and more*/

//the code makes new promise that will be resolved once the image loaded successfully
/*crossOrigin is an img attribute that allows for the loading 
*images across domains, and gets around CORS (cross-origin resource sharing) issues
*when interacting with the DOM.*/

    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';

/*onload event can be used to check visitors browser type and version
*and loads the proper version of webpage based on information*/

        img.onload = () => {

/*natural width and height refers to the original dimensions of the loaded image,
*and makes sure that the image size is correct when performing calculations*/

        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

/*An ARRAY BUFFER object is used to represent fixed-length raw binary data buffer.
*the contents of an array buffer cannot be directly manipulated
*can only be accessed through dataview object.These objects can be
*used to read and write the contents of the buffer.*/
//new buffer is like a container that holds every pixel of an image.
        const datasetBytesBuffer =
            new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);//4 represents number of channels


/*chunksize is the number of rows to be read into a dataframe
* at any time inorder to fit into the local memory*/

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
              datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
              IMAGE_SIZE * chunkSize);

//ctx.drawimage draws image,canvas or video
//canvas is used to draw graphics via js

          ctx.drawImage(
              img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
              chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

//below code loops through every image in the sprite and intializes new array for that iterations

          for (let j = 0; j < imageData.data.length / 4; j++) {

// All channels hold an equal value since the image is grayscale, so
// just read the red channel.

            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }

//below line takes the buffer and recasts into new array that holds our pixel data and resolves our promise.
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
      //this line begins loading the image, which starts the function.
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] =
        await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

// Create shuffled indices into the train/test set for when we select a
// random dataset element for training / validation.

    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

// Slice the the images and labels into train and test sets.
//slice method returns the selected elements in an array,as a new array object

    this.trainImages =
        this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels =
        this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels =
        this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
        batchSize, [this.trainImages, this.trainLabels], () => {
          this.shuffledTrainIndex =
              (this.shuffledTrainIndex + 1) % this.trainIndices.length;
          return this.trainIndices[this.shuffledTrainIndex];
        });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
          (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const image =
          data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label =
          data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return {xs, labels};
  }
}
