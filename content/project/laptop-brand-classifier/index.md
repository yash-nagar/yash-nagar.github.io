---
title: Laptop Brand Classifier
date: 2021-12-06T20:04:28.104Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
<div class="cell markdown" data-colab_type="text" id="VcqRb4QYkmbP">

# **What brand is this laptop?**

Based on Lesson 2 of fast.ai's Deep Learning course, it is possible to
scrape images of the internet (particularly Google Images) to build our
own classifier, which is actually extremely useful and can be applied to
any number of applications.

Here, I chose a really simple problem, to classify laptops based on
their brands using images of them. Although it may not seem so simple,
since all laptops look similar to a certain extent, the highly efficient
Deep Learning models will beg to differ.

This model gets around **83% accuracy**, which is a very good result
considering how similar laptops from different brands look.

This is the code used to carry out this task:

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="gsHHdV6XluGZ">

``` python
from fastai.vision import *
```

</div>

<div class="cell markdown" data-colab_type="text" id="PvUq72LIl39D">

After going on Google Images, and searching for whatever images we want
(e.g Macbooks), we can insert a simple Javascript command into the
browser:

``` javascript
urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
```

</div>

<div class="cell markdown" data-colab_type="text" id="iYdSq-9ymedA">

Next, we create the necessary folder and file name for the data to be
imported into.

I am using Google's Colab so all the images will be stored in Google
Drive, from which the images are easily accesible.

</div>

<div class="cell code" data-execution_count="2" data-colab="{&quot;height&quot;:129,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="T1TN9-WNkY70" data-outputId="5d289c41-99e6-4a39-9f5f-d9e66bb014cc">

``` python
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"
base_dir = root_dir + 'fastai-v3'
```

<div class="output stream stdout">

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive

</div>

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="Ln3ej5gBmtST">

``` python
folder = 'macbook'
file = 'macbook.txt'
```

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="nwWtdIrymwnk">

``` python
folder = 'hp'
file = 'hp.txt'
```

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="Mgc2e9X1mycZ">

``` python
folder = 'lenovo'
file = 'lenovo.txt'
```

</div>

<div class="cell markdown" data-colab_type="text" id="nJjJIf8km0BH">

Code has to be run once for every category.

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="ipCpUf4ym3dJ">

``` python
path = Path(base_dir+'/data/images')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
```

</div>

<div class="cell code" data-execution_count="6" data-colab="{&quot;height&quot;:201,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="ko_JNlxXm5ir" data-outputId="066cbd8c-c4f9-46a2-9148-19bfa76dac5c">

``` python
path.ls()
```

<div class="output execute_result" data-execution_count="6">

    [PosixPath('/content/gdrive/My Drive/fastai-v3/data/images/macbook.txt'),
     PosixPath('/content/gdrive/My Drive/fastai-v3/data/images/macbook'),
     PosixPath('/content/gdrive/My Drive/fastai-v3/data/images/lenovo.txt'),
     PosixPath('/content/gdrive/My Drive/fastai-v3/data/images/hp'),
     PosixPath('/content/gdrive/My Drive/fastai-v3/data/images/hp.txt'),
     PosixPath('/content/gdrive/My Drive/fastai-v3/data/images/lenovo'),
     PosixPath('/content/gdrive/My Drive/fastai-v3/data/images/models'),
     PosixPath('/content/gdrive/My Drive/fastai-v3/data/images/cleaned.csv'),
     PosixPath('/content/gdrive/My Drive/fastai-v3/data/images/export.pkl'),
     PosixPath('/content/gdrive/My Drive/fastai-v3/data/images/mactest.jpg')]

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="GY33yah7nACc">

Next, the files (txt files with urls of images) has to be uploaded into
Drive.

Once that is done, the images can be downloaded into Drive, into the
specified folders, from the urls using the download\_images function.

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="Uo8FOxIMnMph">

``` python
download_images(path/file, dest, max_pics=200)
```

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="oZVxlyo0nSXS">

``` python
classes = ['macbook','hp','lenovo']
```

</div>

<div class="cell markdown" data-colab_type="text" id="3a_JXI-SnPVZ">

We can remove any images that cannot be opened:

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="jCgaZufrnYes">

``` python
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)
```

</div>

<div class="cell markdown" data-colab_type="text" id="39A4szc9nhaE">

Next, we can extract the images from the folders, and seperate them into
training and validation sets, using the ImageDataBunch function.

</div>

<div class="cell code" data-execution_count="9" data-colab="{&quot;height&quot;:90,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="KP5Aa04bne6r" data-outputId="79d39617-dc3b-4b7e-f173-7d577df37afe">

``` python
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
```

<div class="output stream stderr">

    /usr/local/lib/python3.6/dist-packages/fastai/data_block.py:534: UserWarning: You are labelling your items with CategoryList.
    Your valid set contained the following unknown labels, the corresponding items have been discarded.
    images
      if getattr(ds, 'warn', False): warn(ds.warn)

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="oU1-LgL9n2kA">

Looking at some of the pictures:

</div>

<div class="cell code" data-execution_count="10" data-colab="{&quot;height&quot;:568,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="nHrnAOPOn42T" data-outputId="f0de0a93-6775-4a38-e58c-77158e5a3c3f">

``` python
data.show_batch(rows=3, figsize=(7,8))
```

<div class="output display_data">

![](0dd427565e8ff1a01bc0dd4e8f8c7b85820df9fc.png)

</div>

</div>

<div class="cell code" data-execution_count="11" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="-SwN9kbCn-Wt" data-outputId="494f46db-a851-483d-b6cc-7bc24f063562">

``` python
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
```

<div class="output execute_result" data-execution_count="11">

    (['hp', 'lenovo', 'macbook'], 3, 306, 75)

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="4EBj8E24n7Mc">

Training the model, using the cnn\_learner function:

</div>

<div class="cell code" data-execution_count="12" data-colab="{&quot;height&quot;:54,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="f9DQcTYdpB2t" data-outputId="69e26d80-2e19-43a9-f6c1-45c0ec1cc329">

``` python
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
```

<div class="output stream stderr">

    Downloading: "https://download.pytorch.org/models/resnet34-333f7ec4.pth" to /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth
    100%|██████████| 87306240/87306240 [00:00<00:00, 162957184.69it/s]

</div>

</div>

<div class="cell code" data-execution_count="13" data-colab="{&quot;height&quot;:206,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="3sK4XXL1pEDA" data-outputId="75093b66-00d6-4ad6-814e-4c47d4ed46d2">

``` python
learn.fit_one_cycle(5)
```

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

</div>

<div class="cell code" data-execution_count="14" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="_3RnBL1JpzSI" data-outputId="094e0b8a-a3e9-4725-bc56-bddc03aaa74f">

``` python
learn.lr_find(start_lr=1e-5, end_lr=1e-1)
```

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output stream stdout">

    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="tE0vgUX-pVKW">

Interpreting the results:

</div>

<div class="cell code" data-execution_count="15" data-colab="{&quot;height&quot;:283,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="d7duac2op6bF" data-outputId="cf2237b6-d46b-428b-f402-3d1a05cf754c">

``` python
learn.recorder.plot()
```

<div class="output display_data">

![](f05a7178abf5517b21c43b75870dc2d1ae61e739.png)

</div>

</div>

<div class="cell code" data-execution_count="18" data-colab="{&quot;height&quot;:112,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="VLsHSAdXqV7M" data-outputId="79647e14-126c-488c-a81f-5a4fc9f09d39">

``` python
learn.fit_one_cycle(2,max_lr=slice(1e-03,1e-02))
```

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="ZH31vr48pH-c">

``` python
interp = ClassificationInterpretation.from_learner(learn)
```

</div>

<div class="cell code" data-execution_count="20" data-colab="{&quot;height&quot;:311,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="myqzHX6ipbs-" data-outputId="f88ea5c1-9da6-467c-ccb2-5a0690993284">

``` python
interp.plot_confusion_matrix()
```

<div class="output display_data">

![](62b5065a25326ae875403e03cfe5c031aec0fc00.png)

</div>

</div>

<div class="cell code" data-execution_count="21" data-colab="{&quot;height&quot;:90,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="2W81NsN4q_oI" data-outputId="7c1e79fb-57c6-456b-9c88-14e05016cfb5">

``` python
interp.most_confused(min_val=2)
```

<div class="output execute_result" data-execution_count="21">

    [('lenovo', 'hp', 4),
     ('hp', 'macbook', 3),
     ('lenovo', 'macbook', 3),
     ('macbook', 'hp', 3)]

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="MJIyPS5dpgxS">

Lenovo's are being mistaken for HP's 4 times, but the reverse doesn't
seem to happen. Macbooks are the ones that are creating most of the
error.

Using an unused picture, and checking if our model can predict what
laptop brand it is:

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="hhuNeXWSrmll">

``` python
learn.export()
```

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="jQW59finpdzw">

``` python
defaults.device = torch.device('cpu')
```

</div>

<div class="cell code" data-execution_count="30" data-colab="{&quot;height&quot;:254,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="2RHf5dXIrbbc" data-outputId="d16ee440-b9bb-4fe8-c791-7cbb81e9b444">

``` python
img = open_image(path/'mactest.jpg')
img
```

<div class="output execute_result" data-execution_count="30">

![](16bfbb91c9f78f785cab8b695ed1fb2ad4ffe390.jpg)

</div>

</div>

<div class="cell code" data-execution_count="0" data-colab="{}" data-colab_type="code" id="IPcErrtNrdIH">

``` python
learn = load_learner(path)
```

</div>

<div class="cell code" data-execution_count="27" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="C-A8o48Krjdw" data-outputId="97144627-028c-47a7-8ead-b68955d1ca7f">

``` python
pred_class,pred_idx,outputs = learn.predict(img)
pred_class
```

<div class="output execute_result" data-execution_count="27">

    Category macbook

</div>

</div>

<div class="cell code" data-execution_count="29" data-colab="{&quot;height&quot;:365,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="hF8B95YDrvB4" data-outputId="c719880d-c4ff-41fc-ae01-398e3cfcb5a4">

``` python
img1  = open_image(path/'hptest.jpg')
img1
```

<div class="output execute_result" data-execution_count="29">

![](6bef6e7d6581987280a201ad353c234d8f143ee1.jpg)

</div>

</div>

<div class="cell code" data-execution_count="33" data-colab="{&quot;height&quot;:35,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" data-colab_type="code" id="SqWlgPFFsARu" data-outputId="98c707b3-ed32-47d4-ebfd-2c08effd269e">

``` python
pred_class,pred_idx,outputs = learn.predict(img1)
pred_class
```

<div class="output execute_result" data-execution_count="33">

    Category hp

</div>

</div>

<div class="cell markdown" data-colab_type="text" id="73U279aGsJE1">

The model is able to predict these new images perfectly as well.

A very simple application to do something pretty complex.

</div>
