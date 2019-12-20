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

