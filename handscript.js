console.log("hi tf");
import{MnistData} from './data.js';

async function showExamples(data){
    //creating container in the visor
   const surface=tfvis.visor().surface({'name':'Input Handwritten data', tab:'Input data'});
   //get the examples
   const examples=data.nextTestBatch(20);
   const numExamples=examples.xs.shape[0];

   //create a canvas element to render each example
   for(let i=0;i<numExamples;i++){
       const imageTensor=tf.tidy(()=>{
           return examples.xs
           .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
   
   }
}