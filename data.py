import torch
import cv2
from fastai.vision import *
import tifffile as tiff

def choose_random(x):
    "Return a random element from x"
    return np.random.choice(x, 1)[0]

def stats(x):
    "print out dtype, shape, min, max, mean and std of x"
    print(x.dtype)
    print(x.shape)
    print(x.min(), x.max())
    print(x.mean(), x.std())
    
# LITE ON THESE FUNCTIONS. THEY'RE SLOWER THAN SEQUENTIAL
def parallel_imread(fname, index):
    "returns tuple of image and index because order of execution isn't guaranteed in parallel execution"
    im = cv2.imread(fname, cv2.IMREAD_UNCHANGED) / 65535.0
#     im = np.asarray(PIL.Image.open(fname)) / 255.0
    if len(im.shape) == 2:
        im = im[:,:,None]
    return (im, index)

def parallel_open(fnames):
    imlist = parallel(parallel_imread, fnames)
    imlist.sort(key=lambda x:x[1])
    return [o[0] for o in imlist]

def parallel_open_and_stack(fnames):
    im_list = parallel_open(fnames)
    im = np.dstack(im_list)
    return Image(torch.Tensor(im).permute(2, 0, 1))

def get_channel_im_names(fn):
    "Returns a list of all the names of the images associated with a training sample"
    fn = str(fn)
    names = [f'{fn}_{i}.tif' for i in range(10)] + [f'{fn}_{11}.tif']
    return names

def open_and_stack(fnames):
    "Open all images in fnames and stack them in a torch tensor on the GPU"
    im_list = []
    for fname in fnames[:-1]:
        im_list.append(torch.tensor(cv2.imread(fname, cv2.IMREAD_UNCHANGED).astype(np.int16)))
    im_list.append(torch.tensor(cv2.imread(fnames[-1], cv2.IMREAD_UNCHANGED)[:,:,None].astype(np.int16)))
    im = torch.cat(im_list, dim=2).permute(2, 0, 1).float() / 65535.0
    return Image(im)


#  ```python
class HSImageListTIFF(ImageList):
    def open(self, fn):
        t = torch.tensor(tiff.imread(str(fn)).transpose(2, 0, 1)).float()
        return Image(t)
    
    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show the `xs` (inputs) and `ys` (targets) on a figure of `figsize`."
#         raise ValueError()
        rows = int(np.ceil(math.sqrt(len(xs))))
        axs = subplots(rows, rows, imgsize=imgsize, figsize=figsize)
        for x,y,ax in zip(xs, ys, axs.flatten()): Image(x._px[:3]).show(ax=ax, y=y , **kwargs)
        for ax in axs.flatten()[len(xs):]: ax.axis('off')
        plt.tight_layout()

class HSImageList(ImageList):
    "Our simple wrapper for HySi Images"
    def open(self, fn):
        fnames = get_channel_im_names(fn)
        return open_and_stack(fnames)

class ImageHSImageList(ImageList):
    _label_cls,_square_show,_square_show_res = HSImageList,False,False
    _label_cls = HSImageListTIFF
    
    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show the `xs` (inputs) and `ys` (targets) on a figure of `figsize`."
#         raise ValueError()
        rows = int(np.ceil(math.sqrt(len(xs))))
        axs = subplots(rows, rows, imgsize=imgsize, figsize=figsize)
        for x,y,ax in zip(xs, ys, axs.flatten()): x.show(ax=ax, y=Image(y._px[:3]) , **kwargs)
        for ax in axs.flatten()[len(xs):]: ax.axis('off')
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
        if self._square_show_res:
            title = 'Ground truth\nPredictions'
            rows = int(np.ceil(math.sqrt(len(xs))))
            axs = subplots(rows, rows, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=12)
            for x,y,z,ax in zip(xs,ys,zs,axs.flatten()): x.show(ax=ax, title=f'{str(y)}\n{str(z)}', **kwargs)
            for ax in axs.flatten()[len(xs):]: ax.axis('off')
        else:
            title = 'Ground truth/Predictions'
            axs = subplots(len(xs), 2, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=14)
            for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
                x.show(ax=axs[i,0], y=Image(y._px[:3]), **kwargs)
                x.show(ax=axs[i,1], y=Image(z._px[:3]), **kwargs)
                

# labeller = (lambda x: f'../Data/Train_Spectral_Images/{x.name[:-10]}')

def get_data(bs=8, size="full", pct=0.1, dataset='new', tfms=None, PATH='../Data/Train_Clean', labeller=None, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    labeller = ifnone(labeller, (lambda x: f"../Data/Train_Spectral_Tensors/{x.stem[:-6] + '.tif'}"))
    #tfms = ifnone(tfms, get_transforms(max_warp=0.0))
    train_data = ImageHSImageList.from_folder(path=PATH,extensions='.png')
    if dataset == 'full':
        pass
    elif dataset == 'new' or dataset == 'pseudo':
        train_data = train_data.filter_by_func(lambda p: "ARAD" in str(p))
    elif dataset == 'old':
        train_data = train_data.filter_by_func(lambda p: "BGU" in str(p))
    if dataset=='pseudo':
        train_data.add(ImageHSImageList.from_folder(path='../Data/Pseudo_Clean',extensions='.png'))
    # valid_data = ImageHSImageList.from_folder(path='Validation_Clean',extensions='.png')
    data = train_data.split_by_rand_pct(pct)
    data = data.label_from_func(labeller)
    if size=="full":
        data = data.transform(tfms, tfm_y=True).databunch(bs=bs).normalize()
    else:
        data = data.transform(tfms, size=size, tfm_y=True).databunch(bs=bs).normalize()
    data.c = 31
    return data

def labeller_valid(x):
    fname = '_'.join(x.stem.split('_')[:-1]) + '.tif'
    if 'Validation' in x.parent.name:
        parent = Path('../Data/Validation_Spectral_Tensors')
    elif 'Pseudo' in x.parent.name:
        parent = Path('../Data/Pseudo_Spectral_Tensors')
    elif 'Test' in x.parent.name:
        parent = Path('../Data/Pseudo_Spectral_Tensors')
    else:
        parent = Path('../Data/Train_Spectral_Tensors')
    return parent/fname

def splitter_valid(x):
    return 'Validation' in x.parent.name

real_stats = (tensor([0.1859, 0.1794, 0.1533]), tensor([0.1244, 0.1235, 0.1181]))
clean_stats = (tensor([0.1801, 0.1821, 0.1468]), tensor([0.1220, 0.1245, 0.1171]))

def get_data_new(bs=8, size="full", dataset='new', tfms=None, PATH='../Data/', labeller=labeller_valid, 
                 splitter=splitter_valid, folders=None, seed=42, stats=clean_stats):
    np.random.seed(seed)
    torch.manual_seed(seed)

    folders = ifnone(folders, ['Train_Clean', 'Validation_Clean'])
    train_data = ImageHSImageList.from_folder(path=PATH, include=folders, extensions=['.png','.jpg'])
    # Filter out images from last year's dataset
    if dataset == 'full':
        pass
    elif dataset == 'new' or dataset == 'pseudo':
        train_data = train_data.filter_by_func(lambda p: "ARAD" in str(p))
    elif dataset == 'old':
        train_data = train_data.filter_by_func(lambda p: "BGU" in str(p))
    if dataset=='pseudo':
        train_data.add(ImageHSImageList.from_folder(path='../Data/Pseudo_Clean',extensions=['.png','.jpg']))
        train_data.filter_by_func(lambda p: "ARAD" in str(p))
    # Split and label every image
    data = train_data.split_by_valid_func(splitter)
    data = data.label_from_func(labeller)
    # Transformations
    if size=="full":
        data = data.transform(tfms, tfm_y=True).databunch(bs=bs).normalize(stats)
    else:
        data = data.transform(tfms, size=size, tfm_y=True).databunch(bs=bs).normalize(stats)
    # Make sure number of classes is right
    data.c = 31
    return data