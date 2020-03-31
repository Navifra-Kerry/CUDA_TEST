#include <iostream>
#include "vision_cuda.h"
#include <torch/torch.h>

int main()
{
    torch::Tensor t = torch::rand({ 224,224,3 });
    //아래 코드는 동작안함 메트릭스 구조가 다르기때문에 호출이 가능 하다는 예시로 생각하세요.
    ROIPool_forward_cuda(t, t, 0.7, 7, 7);
}