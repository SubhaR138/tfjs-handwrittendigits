console.log("hi tf");


//import {MnistData} from './data.js';

 
async function showExamples(data){

//creating container in the visor
/*visor returns the singleton instance of visor class
*singleton is a class that allows only single instance of itself to be created */

   const surface=tfvis.visor().surface({'name':'Input Handwritten data', tab:'Input data'});

//get the examples

   const examples=data.nextTestBatch(20);
   const numExamples=examples.xs.shape[0];
//xs reperesents a tensor or a list of tensors

//create a canvas element to render each example

   for(let i=0;i<numExamples;i++){

//tf.tidy prevents the memory leakage

       const imageTensor=tf.tidy(()=>{
           return examples.xs
           .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
//canvas is an imagedata with only 0 values
    const canvas=document.createElement('canvas');
    canvas.height=28;
    canvas.width=28;
    canvas.style='margin:4px;';
//tf.browser.toPixels draws a tf.tensor of pixel values to byte array
//appendChild method appends the node as a last child of a node
await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();

   
   }
}
async function run(){
  const data = new MnistData();
  await data.load();
  await showExamples(data);
}

document.addEventListener('DOMContentLoaded', run);
/*DOM content loaded is the point when both DOM is ready and
*there are no style sheets blocking js execution.
*Document Content Object is an API for HTML and XML docs
*It defines the logical structure of docs*/

//define the model architecture

function getModel(){
    const model=tf.sequential();

//Each image is 28x28 in size and have 1 color channel as a grayscale image.
    const IMAGE_WIDTH=28;
    const IMAGE_HEIGHT=28;
    const IMAGE_CHANNELS=1;

//here we are adding conv2d layer instead of fully connected layers
//kernels are small input matrix used in cnn to extract features,which refers to the width height of the filter mask
//filter size refers to the dimension of the filter in cnn
//strides define the number of pixels shift over the input matrix.
//kernel initializing method is for randomly initializing the model weights
model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
/*Usually pooling will be used to remove the negative values
*maxpooling will reduce the amount of computations
*will sent only important data to the next layer
*takes only larger value pixels among all pixels*/

model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}))

//we are adding more filters because cnn wil have number of filters

model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
//flatten layer converts 2d image to 1d
  model.add(tf.layers.flatten());
  const NUM_OUTPUT_CLASSES = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,

//Categoricalcrossentropy loss is used because here we are using 10 classes
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;


}
