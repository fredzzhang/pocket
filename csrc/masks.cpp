#include <torch/extension.h>
#include <math.h>

torch::Tensor GenerateMasks(torch::Tensor boxes, uint h, uint w){
  // Assert boxes is a 2-d tensor of type float
  auto boxes_acc = boxes.accessor<float,2>();
  uint numBoxes = boxes_acc.size(0);
  // Create empty masks
  torch::Tensor masks = torch::zeros({numBoxes, h, w});

  for (i = 0; i < numBoxes; i++) {
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
    float x1_idx = floor(x1); float y1_idx = floor(y1);
    float x2_idx = ceil(x2-1); float y2_idx = ceil(y2-2);
    for (j = (int) y1_idx, j <= (int) y2_idx; j++) {
      for (k = (int) x1_idx, k <= (int) x2_idx; k++) {
        torch::Tensor val = torch::ones(1);
        if (j == y1_idx) {
          val *= (1 + x1_idx - x1);
        } else if (j == y2_idx) {
          val *= (x2 - x2_idx);
        }
        if (k == x1_idx) {
          val *= (1 + y1_idx - y1);
        } else if (k == x2_idx) {
          val *= (y1 - y1_idx);
        }
        masks[i][j][k] = val;
      }
    }
  }
  return masks;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.doc() = "Generate binary masks for a bounding box";
  m.def("generate_masks", &GenerateMasks, "Binary mask generation");
}