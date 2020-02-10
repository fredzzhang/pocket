/*
 * Generate binary masks based on bounding box coordinates
 *
 * Fred Zhang <frederic.zhang@anu.edu.au>
 *
 * The Australian National University
 * Australian Centre for Robotic Vision
*/

#include <torch/extension.h>
#include <math.h>

torch::Tensor GenerateMasks(torch::Tensor boxes, uint h, uint w){
   // Assert boxes is a 2-d tensor of type float
   auto boxes_acc = boxes.accessor<float,2>();
   uint numBoxes = boxes_acc.size(0);
   // Create empty masks
   torch::Tensor masks = torch::zeros({numBoxes, h, w});

   for (uint i = 0; i < numBoxes; i++) {
      float x1 = boxes_acc[i][0]; float y1 = boxes_acc[i][1];
      float x2 = boxes_acc[i][2]; float y2 = boxes_acc[i][3];
      /*
      This is crucial to understand

      1. For a pixel, the coordinates of the point at its top left
         corner give its index
      2. For a point (x,y) within a pixel (not on the border), the 
         index of the pixel is always (floor(x), floor(y))
      3. For a bounding box defined as (x1, y1, x2, y2), the pixel
         that contains the top left corner can be indexed as 
            (floor(x1), floor(y1))
         But the pixel that contains the bottom right coner must be
         indexed as
            (ceil(x2-1), ceil(y2-1)) NOT (floor(x2), floor(y2))
      */
      float x1_f = floor(x1); float y1_f = floor(y1);
      float x2_c = ceil(x2); float y2_c = ceil(y2);

      masks[i].slice(0, (int)y1_f, (int)y2_c).slice(1, (int)x1_f, (int)x2_c) = 1;

      if (y1_f == y2_c - 1) {
         masks[i][(int)y1_f].slice(0, (int)x1_f, (int)x2_c) *= 
            (y2 - y1);
      } else {
         masks[i][(int)y1_f].slice(0, (int)x1_f, (int)x2_c) *= 
            (1 + y1_f - y1);
         masks[i][(int)y2_c - 1].slice(0, (int)x1_f, (int)x2_c) *= 
            (1 + y2 - y2_c);
      }
      if (x1_f == x2_c - 1) {
         masks[i].select(1, (int)x1_f).slice(0, (int)y1_f, (int)y2_c) *=
            (x2 - x1);
      } else {
         masks[i].select(1, (int)x1_f).slice(0, (int)y1_f, (int)y2_c) *= 
            (1 + x1_f - x1);
         masks[i].select(1, (int)x2_c - 1).slice(0, (int)y1_f, (int)y2_c) *= 
            (1 + x2 - x2_c);
      } 

   }
   return masks;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("generate_masks", &GenerateMasks, "Binary mask generation");
}
