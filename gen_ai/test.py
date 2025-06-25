import scipy
print('scipy: %s' % scipy.__version__)
import numpy
print('numpy: %s ' % numpy.__version__)
# import matplotlib
# print('matplotlib: %s ' % matplotlib.__version__)
import sklearn
print('sklearn: %s' % sklearn.__version__)
# import pandas
# print('pandas: %s' % pandas.__version__)
import torch
print("pytorch: v",torch.__version__ , "cuda:", torch.version.cuda, "cudnn:", torch.backends.cudnn.version())
x=torch.rand(5,3)
print(x)
t1 =torch.IntTensor([1, 2, 3])
print(t1)
y=x.cuda()
print(y)

if(torch.cuda.is_available()):
    print(f"cuda available:{torch.cuda.device_count()}" )
    for i in range(torch.cuda.device_count()):
        print(f"cuda device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Cuda is NOT available.")

heights = [189, 170, 189, 163, 183, 171, 185,
168, 173, 183, 173, 173, 175, 178,
183, 193, 178, 173, 174, 183, 183,
180, 168, 180, 170, 178, 182, 180,
183, 178, 182, 188, 175, 179, 183,
193, 182, 183, 177, 185, 188, 188,
182, 185, 191, 183]

height_tensor = torch.tensor(heights, dtype=torch.float64)

print(height_tensor)