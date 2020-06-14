
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a3546411d6bd7687853939e80cbec00c7ee05d3b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'a3546411d6bd7687853939e80cbec00c7ee05d3b', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a3546411d6bd7687853939e80cbec00c7ee05d3b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a3546411d6bd7687853939e80cbec00c7ee05d3b

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/a3546411d6bd7687853939e80cbec00c7ee05d3b

 ************************************************************************************************************************

  ############Check model ################################ 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/ag_news_csv",
    "train":true,
    "alphabet_size":69,
    "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size":1014,
    "num_of_classes":4
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "colX":[
          "colX"
        ],
        "coly":[
          "coly"
        ],
        "encoding":"'ISO-8859-1'",
        "read_csv_parm":{
          "usecols":[
            0,
            1
          ],
          "names":[
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name":"tokenizer",
      "uri":"mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args":{
        "data_source":"",
        "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size":1014,
        "num_of_classes":4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/ag_news_csv",
    "train":true,
    "alphabet_size":69,
    "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size":1014,
    "num_of_classes":4
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "colX":[
          "colX"
        ],
        "coly":[
          "coly"
        ],
        "encoding":"'ISO-8859-1'",
        "read_csv_parm":{
          "usecols":[
            0,
            1
          ],
          "names":[
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name":"tokenizer",
      "uri":"mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args":{
        "data_source":"",
        "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size":1014,
        "num_of_classes":4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.json [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/imdb",
    "pass_data_pars":false,
    "train":true,
    "maxlen":40,
    "max_features":5
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:NumpyDataset",
      "args":{
        "numpy_loader_args":{
          "allow_pickle":true
        },
        "encoding":"'ISO-8859-1'"
      }
    },
    {
      "name":"imdb_process",
      "uri":"mlmodels.preprocess.text_keras:IMDBDataset",
      "args":{
        "num_words":5
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:NumpyDataset {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.NumpyDataset'> 
cls_name : NumpyDataset
Dataset File path :  train/mlmodels/dataset/text/imdb.npz
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.json [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/text\/",
    "dataset":"ner_dataset.csv",
    "pass_data_pars":false,
    "train":true
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "read_csv_parm":{
          "encoding":"ISO-8859-1"
        },
        "colX":[

        ],
        "coly":[

        ]
      }
    },
    {
      "uri":"mlmodels.preprocess.text_keras:Preprocess_namentity",
      "args":{
        "max_len":75
      },
      "internal_states":[
        "word_count"
      ]
    },
    {
      "name":"split_xy",
      "uri":"mlmodels.dataloader:split_xy_from_dict",
      "args":{
        "col_Xinput":[
          "X"
        ],
        "col_yinput":[
          "y"
        ]
      }
    },
    {
      "name":"split_train_test",
      "uri":"sklearn.model_selection:train_test_split",
      "args":{
        "test_size":0.5
      }
    },
    {
      "name":"saver",
      "uri":"mlmodels.dataloader:pickle_dump",
      "args":{
        "path":"mlmodels\/ztest\/ml_keras\/namentity_crm_bilstm\/data.pkl"
      }
    }
  ],
  "output":{
    "shape":[
      [
        75
      ],
      [
        75,
        18
      ]
    ],
    "max_len":75
  }
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'read_csv_parm': {'encoding': 'ISO-8859-1'}, 'colX': [], 'coly': []} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset

  URL:  mlmodels.preprocess.text_keras:Preprocess_namentity {'max_len': 75} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.text_keras.Preprocess_namentity'> 
cls_name : Preprocess_namentity

  
 Object Creation 

  
 Object Compute 

  
 Object get_data 

  URL:  mlmodels.dataloader:split_xy_from_dict {'col_Xinput': ['X'], 'col_yinput': ['y']} 

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f0039c3d620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f0039c3d620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f00a52040d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f00a52040d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f00bef55ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f00bef55ea0> 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/torchhub_cnn_dataloader.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/MNIST",
    "dataset":"MNIST",
    "data_type":"tch_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tch_dataset_start",
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"torchvision.datasets:MNIST",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars":false,
          "arg":{
            "fixed_size":256,
            "path":"dataset\/vision\/MNIST\/"
          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0052a80950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0052a80950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0052a80950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 502, in __init__
    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 502, in __init__
    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 607, in __init__
    data            = np.load( file_path,**args.get("numpy_loader_args", {}))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
    fid = open(os_fspath(file), "rb")
FileNotFoundError: [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 301, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 93, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:04, 153134.98it/s] 50%|█████     | 4972544/9912422 [00:00<00:22, 218471.82it/s]9920512it [00:00, 37306723.04it/s]                           
0it [00:00, ?it/s]32768it [00:00, 588185.50it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 484308.32it/s]1654784it [00:00, 12143650.74it/s]                         
0it [00:00, ?it/s]8192it [00:00, 200536.59it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f004fcad278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f004fe97fd0>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/model_list_CIFAR.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/cifar10\/",
    "dataset":"CIFAR10",
    "data_type":"tf_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tf_dataset_start",
      "uri":"mlmodels.preprocess.generic:tf_dataset_download",
      "arg":{
        "train_samples":2000,
        "test_samples":500,
        "shuffle":true,
        "download":true
      }
    },
    {
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"mlmodels.preprocess.generic:NumpyDataset",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_generic",
          "pass_data_pars":false,
          "arg":{
            "fixed_size":256
          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:tf_dataset_download {} 

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f0052a80598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f0052a80598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f0052a80598> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<00:59,  2.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:59,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:58,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:58,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:58,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:57,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:57,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:40,  3.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:40,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:40,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:40,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:40,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:39,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:39,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:39,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:27,  5.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:27,  5.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:27,  5.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:27,  5.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:27,  5.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:18,  7.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:12, 10.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:12, 10.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:12, 10.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:12, 10.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12, 10.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:12, 10.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:12, 10.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12, 10.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:12, 10.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:00<00:08, 13.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:08, 13.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 13.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 13.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 13.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:08, 13.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:08, 13.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 13.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 18.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 18.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 18.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 18.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 18.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 18.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 18.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 23.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 23.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 23.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 23.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 23.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 23.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 23.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 23.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 29.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 29.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 29.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 29.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 29.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 29.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 29.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 29.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 29.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 36.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 36.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 36.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 36.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 36.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 36.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 36.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 36.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 36.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 41.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 41.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 41.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 41.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 41.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 41.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 41.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 41.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 41.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 46.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 46.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 46.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 46.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 46.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 46.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 46.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 52.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 52.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 52.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 52.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 52.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 52.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 52.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 52.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 52.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 56.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 56.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 60.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 63.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 63.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 63.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 63.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 63.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 63.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 63.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 63.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 69.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 69.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 72.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 72.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 72.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 72.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 72.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 72.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 72.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 72.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.57s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.57s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.57s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.32 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.97s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.57s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 75.32 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.97s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.97s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.59 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.97s/ url]
0 examples [00:00, ? examples/s]2020-06-14 15:29:20.323904: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-14 15:29:20.329246: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-14 15:29:20.329430: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56015ef9d320 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-14 15:29:20.329446: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
52 examples [00:00, 518.79 examples/s]147 examples [00:00, 599.50 examples/s]244 examples [00:00, 676.32 examples/s]344 examples [00:00, 748.47 examples/s]432 examples [00:00, 782.61 examples/s]531 examples [00:00, 833.74 examples/s]628 examples [00:00, 869.46 examples/s]725 examples [00:00, 896.28 examples/s]819 examples [00:00, 907.28 examples/s]915 examples [00:01, 920.97 examples/s]1013 examples [00:01, 935.78 examples/s]1115 examples [00:01, 958.77 examples/s]1217 examples [00:01, 973.32 examples/s]1315 examples [00:01, 972.53 examples/s]1413 examples [00:01, 967.54 examples/s]1513 examples [00:01, 975.45 examples/s]1612 examples [00:01, 977.59 examples/s]1710 examples [00:01, 954.08 examples/s]1806 examples [00:01, 940.17 examples/s]1901 examples [00:02, 929.37 examples/s]2000 examples [00:02, 944.17 examples/s]2096 examples [00:02, 946.71 examples/s]2193 examples [00:02, 953.06 examples/s]2291 examples [00:02, 960.21 examples/s]2388 examples [00:02, 955.47 examples/s]2484 examples [00:02, 956.01 examples/s]2580 examples [00:02, 947.26 examples/s]2675 examples [00:02, 932.46 examples/s]2769 examples [00:02, 905.69 examples/s]2861 examples [00:03, 909.81 examples/s]2958 examples [00:03, 924.62 examples/s]3051 examples [00:03, 921.38 examples/s]3147 examples [00:03, 931.28 examples/s]3248 examples [00:03, 951.28 examples/s]3344 examples [00:03, 915.43 examples/s]3440 examples [00:03, 927.06 examples/s]3536 examples [00:03, 934.51 examples/s]3630 examples [00:03, 908.31 examples/s]3722 examples [00:03, 888.54 examples/s]3812 examples [00:04, 868.37 examples/s]3909 examples [00:04, 895.08 examples/s]4003 examples [00:04, 906.88 examples/s]4106 examples [00:04, 939.40 examples/s]4201 examples [00:04, 941.56 examples/s]4296 examples [00:04, 941.48 examples/s]4393 examples [00:04, 948.39 examples/s]4489 examples [00:04, 941.55 examples/s]4588 examples [00:04, 954.88 examples/s]4684 examples [00:05, 953.45 examples/s]4780 examples [00:05, 951.08 examples/s]4876 examples [00:05, 925.16 examples/s]4969 examples [00:05, 913.92 examples/s]5064 examples [00:05, 923.02 examples/s]5160 examples [00:05, 931.60 examples/s]5258 examples [00:05, 945.22 examples/s]5353 examples [00:05, 945.89 examples/s]5448 examples [00:05, 936.46 examples/s]5542 examples [00:05, 937.38 examples/s]5636 examples [00:06, 923.15 examples/s]5729 examples [00:06, 923.62 examples/s]5824 examples [00:06, 931.31 examples/s]5920 examples [00:06, 938.47 examples/s]6018 examples [00:06, 949.56 examples/s]6117 examples [00:06, 958.27 examples/s]6214 examples [00:06, 959.50 examples/s]6313 examples [00:06, 967.84 examples/s]6410 examples [00:06, 956.12 examples/s]6506 examples [00:06, 948.97 examples/s]6603 examples [00:07, 953.20 examples/s]6699 examples [00:07, 953.59 examples/s]6796 examples [00:07, 958.06 examples/s]6892 examples [00:07, 957.01 examples/s]6991 examples [00:07, 964.06 examples/s]7089 examples [00:07, 968.54 examples/s]7186 examples [00:07, 959.35 examples/s]7283 examples [00:07, 960.18 examples/s]7380 examples [00:07, 960.27 examples/s]7478 examples [00:07, 966.02 examples/s]7575 examples [00:08, 963.48 examples/s]7672 examples [00:08, 940.02 examples/s]7771 examples [00:08, 954.13 examples/s]7871 examples [00:08, 964.84 examples/s]7969 examples [00:08, 968.03 examples/s]8066 examples [00:08, 966.87 examples/s]8165 examples [00:08, 970.80 examples/s]8263 examples [00:08, 961.27 examples/s]8360 examples [00:08, 962.54 examples/s]8458 examples [00:08, 967.35 examples/s]8555 examples [00:09, 937.04 examples/s]8653 examples [00:09, 948.67 examples/s]8750 examples [00:09, 952.07 examples/s]8846 examples [00:09, 941.34 examples/s]8949 examples [00:09, 965.98 examples/s]9046 examples [00:09, 956.79 examples/s]9142 examples [00:09, 957.46 examples/s]9238 examples [00:09, 950.08 examples/s]9334 examples [00:09, 948.47 examples/s]9435 examples [00:10, 964.79 examples/s]9532 examples [00:10, 960.82 examples/s]9631 examples [00:10, 967.97 examples/s]9730 examples [00:10, 971.55 examples/s]9828 examples [00:10, 965.57 examples/s]9925 examples [00:10, 960.56 examples/s]10022 examples [00:10, 861.88 examples/s]10119 examples [00:10, 890.60 examples/s]10217 examples [00:10, 913.24 examples/s]10310 examples [00:10, 913.50 examples/s]10407 examples [00:11, 928.94 examples/s]10501 examples [00:11, 927.73 examples/s]10597 examples [00:11, 935.79 examples/s]10691 examples [00:11, 915.91 examples/s]10786 examples [00:11, 923.96 examples/s]10882 examples [00:11, 933.77 examples/s]10977 examples [00:11, 937.22 examples/s]11072 examples [00:11, 940.04 examples/s]11167 examples [00:11, 936.83 examples/s]11261 examples [00:11, 929.59 examples/s]11357 examples [00:12, 936.84 examples/s]11451 examples [00:12, 928.06 examples/s]11551 examples [00:12, 945.92 examples/s]11651 examples [00:12, 959.40 examples/s]11750 examples [00:12, 964.88 examples/s]11848 examples [00:12, 968.01 examples/s]11945 examples [00:12, 950.96 examples/s]12041 examples [00:12, 952.73 examples/s]12137 examples [00:12, 954.89 examples/s]12235 examples [00:12, 959.36 examples/s]12331 examples [00:13, 944.43 examples/s]12427 examples [00:13, 946.90 examples/s]12526 examples [00:13, 957.82 examples/s]12622 examples [00:13, 955.27 examples/s]12718 examples [00:13, 953.60 examples/s]12814 examples [00:13, 933.58 examples/s]12908 examples [00:13, 931.95 examples/s]13003 examples [00:13, 934.90 examples/s]13098 examples [00:13, 939.09 examples/s]13198 examples [00:14, 955.56 examples/s]13294 examples [00:14, 955.26 examples/s]13395 examples [00:14, 969.63 examples/s]13495 examples [00:14, 976.91 examples/s]13597 examples [00:14, 987.09 examples/s]13700 examples [00:14, 997.15 examples/s]13800 examples [00:14, 963.76 examples/s]13897 examples [00:14, 954.08 examples/s]13993 examples [00:14, 951.48 examples/s]14089 examples [00:14, 951.75 examples/s]14186 examples [00:15, 956.83 examples/s]14283 examples [00:15, 958.43 examples/s]14379 examples [00:15, 903.45 examples/s]14475 examples [00:15, 918.63 examples/s]14570 examples [00:15, 927.68 examples/s]14664 examples [00:15, 925.57 examples/s]14757 examples [00:15, 915.47 examples/s]14850 examples [00:15, 918.28 examples/s]14945 examples [00:15, 925.80 examples/s]15040 examples [00:15, 932.68 examples/s]15134 examples [00:16, 929.91 examples/s]15228 examples [00:16, 930.95 examples/s]15324 examples [00:16, 937.32 examples/s]15421 examples [00:16, 945.77 examples/s]15516 examples [00:16, 932.01 examples/s]15610 examples [00:16, 921.91 examples/s]15710 examples [00:16, 941.88 examples/s]15805 examples [00:16, 944.12 examples/s]15900 examples [00:16, 920.59 examples/s]15993 examples [00:16, 923.03 examples/s]16087 examples [00:17, 927.01 examples/s]16184 examples [00:17, 939.04 examples/s]16279 examples [00:17, 928.47 examples/s]16378 examples [00:17, 942.78 examples/s]16474 examples [00:17, 944.64 examples/s]16569 examples [00:17, 936.96 examples/s]16663 examples [00:17, 905.12 examples/s]16757 examples [00:17, 913.48 examples/s]16849 examples [00:17, 913.62 examples/s]16943 examples [00:18, 921.07 examples/s]17039 examples [00:18, 931.63 examples/s]17137 examples [00:18, 945.14 examples/s]17232 examples [00:18, 905.26 examples/s]17324 examples [00:18, 908.77 examples/s]17416 examples [00:18, 905.88 examples/s]17507 examples [00:18, 906.89 examples/s]17607 examples [00:18, 931.79 examples/s]17702 examples [00:18, 934.57 examples/s]17800 examples [00:18, 945.47 examples/s]17898 examples [00:19, 953.60 examples/s]17994 examples [00:19, 949.86 examples/s]18092 examples [00:19, 957.13 examples/s]18188 examples [00:19, 956.15 examples/s]18285 examples [00:19, 960.05 examples/s]18388 examples [00:19, 977.89 examples/s]18486 examples [00:19, 971.39 examples/s]18584 examples [00:19, 957.26 examples/s]18680 examples [00:19, 951.43 examples/s]18776 examples [00:19, 931.98 examples/s]18876 examples [00:20, 949.46 examples/s]18972 examples [00:20, 942.85 examples/s]19067 examples [00:20, 938.97 examples/s]19162 examples [00:20, 939.29 examples/s]19256 examples [00:20, 922.03 examples/s]19354 examples [00:20, 936.36 examples/s]19451 examples [00:20, 943.63 examples/s]19548 examples [00:20, 950.43 examples/s]19647 examples [00:20, 961.67 examples/s]19745 examples [00:20, 966.59 examples/s]19844 examples [00:21, 973.33 examples/s]19943 examples [00:21, 976.84 examples/s]20041 examples [00:21, 864.45 examples/s]20140 examples [00:21, 895.09 examples/s]20238 examples [00:21, 918.05 examples/s]20334 examples [00:21, 930.20 examples/s]20434 examples [00:21, 948.47 examples/s]20530 examples [00:21, 948.79 examples/s]20630 examples [00:21, 961.01 examples/s]20727 examples [00:22, 962.04 examples/s]20825 examples [00:22, 964.52 examples/s]20924 examples [00:22, 969.52 examples/s]21022 examples [00:22, 959.15 examples/s]21119 examples [00:22, 960.72 examples/s]21217 examples [00:22, 966.01 examples/s]21317 examples [00:22, 973.20 examples/s]21415 examples [00:22, 966.26 examples/s]21512 examples [00:22, 956.19 examples/s]21611 examples [00:22, 964.54 examples/s]21708 examples [00:23, 958.42 examples/s]21806 examples [00:23, 962.04 examples/s]21907 examples [00:23, 975.64 examples/s]22005 examples [00:23, 972.93 examples/s]22103 examples [00:23, 967.75 examples/s]22200 examples [00:23, 957.05 examples/s]22296 examples [00:23, 951.67 examples/s]22394 examples [00:23, 959.21 examples/s]22490 examples [00:23, 958.30 examples/s]22589 examples [00:23, 966.72 examples/s]22686 examples [00:24, 964.25 examples/s]22783 examples [00:24, 955.51 examples/s]22879 examples [00:24, 955.91 examples/s]22975 examples [00:24, 895.79 examples/s]23073 examples [00:24, 918.01 examples/s]23166 examples [00:24, 914.18 examples/s]23262 examples [00:24, 927.26 examples/s]23361 examples [00:24, 944.83 examples/s]23456 examples [00:24, 937.87 examples/s]23554 examples [00:24, 950.04 examples/s]23652 examples [00:25, 956.19 examples/s]23748 examples [00:25, 927.00 examples/s]23842 examples [00:25, 921.80 examples/s]23939 examples [00:25, 934.05 examples/s]24039 examples [00:25, 952.89 examples/s]24138 examples [00:25, 962.22 examples/s]24237 examples [00:25, 967.31 examples/s]24338 examples [00:25, 977.34 examples/s]24437 examples [00:25, 979.19 examples/s]24537 examples [00:25, 984.01 examples/s]24636 examples [00:26, 972.15 examples/s]24734 examples [00:26, 962.53 examples/s]24831 examples [00:26, 953.06 examples/s]24927 examples [00:26, 938.07 examples/s]25024 examples [00:26, 945.83 examples/s]25123 examples [00:26, 956.74 examples/s]25221 examples [00:26, 962.46 examples/s]25318 examples [00:26, 953.34 examples/s]25414 examples [00:26, 945.25 examples/s]25512 examples [00:27, 953.95 examples/s]25608 examples [00:27, 951.37 examples/s]25705 examples [00:27, 955.86 examples/s]25803 examples [00:27, 961.86 examples/s]25900 examples [00:27, 927.90 examples/s]26001 examples [00:27, 948.62 examples/s]26097 examples [00:27, 949.87 examples/s]26193 examples [00:27, 951.75 examples/s]26289 examples [00:27, 942.25 examples/s]26385 examples [00:27, 947.13 examples/s]26481 examples [00:28, 950.41 examples/s]26577 examples [00:28, 948.34 examples/s]26672 examples [00:28, 944.34 examples/s]26771 examples [00:28, 955.40 examples/s]26867 examples [00:28, 948.82 examples/s]26962 examples [00:28, 920.98 examples/s]27056 examples [00:28, 926.06 examples/s]27154 examples [00:28, 941.21 examples/s]27250 examples [00:28, 944.68 examples/s]27347 examples [00:28, 950.79 examples/s]27445 examples [00:29, 959.12 examples/s]27541 examples [00:29, 959.32 examples/s]27637 examples [00:29, 955.78 examples/s]27733 examples [00:29, 935.06 examples/s]27829 examples [00:29, 940.51 examples/s]27927 examples [00:29, 949.49 examples/s]28023 examples [00:29, 945.28 examples/s]28118 examples [00:29, 941.55 examples/s]28213 examples [00:29, 928.29 examples/s]28311 examples [00:29, 941.65 examples/s]28406 examples [00:30, 932.61 examples/s]28503 examples [00:30, 941.19 examples/s]28598 examples [00:30, 943.63 examples/s]28693 examples [00:30, 934.77 examples/s]28787 examples [00:30, 915.49 examples/s]28886 examples [00:30, 935.14 examples/s]28988 examples [00:30, 957.85 examples/s]29085 examples [00:30, 957.85 examples/s]29181 examples [00:30, 953.64 examples/s]29281 examples [00:31, 966.03 examples/s]29381 examples [00:31, 974.92 examples/s]29479 examples [00:31, 973.73 examples/s]29579 examples [00:31, 981.10 examples/s]29678 examples [00:31, 971.81 examples/s]29776 examples [00:31, 969.68 examples/s]29874 examples [00:31, 966.54 examples/s]29974 examples [00:31, 972.44 examples/s]30072 examples [00:31, 873.69 examples/s]30162 examples [00:31, 872.97 examples/s]30262 examples [00:32, 906.20 examples/s]30359 examples [00:32, 922.22 examples/s]30455 examples [00:32, 932.15 examples/s]30549 examples [00:32, 933.26 examples/s]30644 examples [00:32, 936.71 examples/s]30741 examples [00:32, 944.96 examples/s]30836 examples [00:32, 944.31 examples/s]30933 examples [00:32, 950.49 examples/s]31031 examples [00:32, 958.22 examples/s]31127 examples [00:32, 945.25 examples/s]31229 examples [00:33, 964.78 examples/s]31328 examples [00:33, 972.00 examples/s]31427 examples [00:33, 975.80 examples/s]31526 examples [00:33, 979.02 examples/s]31624 examples [00:33, 971.93 examples/s]31722 examples [00:33, 940.37 examples/s]31817 examples [00:33, 927.21 examples/s]31915 examples [00:33, 940.85 examples/s]32011 examples [00:33, 946.36 examples/s]32106 examples [00:34, 924.46 examples/s]32200 examples [00:34, 928.29 examples/s]32296 examples [00:34, 937.57 examples/s]32393 examples [00:34, 946.71 examples/s]32492 examples [00:34, 957.67 examples/s]32588 examples [00:34, 956.32 examples/s]32684 examples [00:34, 948.51 examples/s]32782 examples [00:34, 956.98 examples/s]32883 examples [00:34, 970.27 examples/s]32982 examples [00:34, 974.93 examples/s]33080 examples [00:35, 958.48 examples/s]33176 examples [00:35, 950.93 examples/s]33272 examples [00:35, 936.90 examples/s]33369 examples [00:35, 944.87 examples/s]33466 examples [00:35, 952.06 examples/s]33562 examples [00:35, 943.13 examples/s]33661 examples [00:35, 956.27 examples/s]33757 examples [00:35, 953.81 examples/s]33853 examples [00:35, 954.50 examples/s]33949 examples [00:35, 955.53 examples/s]34045 examples [00:36, 954.66 examples/s]34142 examples [00:36, 957.73 examples/s]34240 examples [00:36, 961.63 examples/s]34337 examples [00:36, 961.69 examples/s]34434 examples [00:36, 963.26 examples/s]34531 examples [00:36, 958.19 examples/s]34627 examples [00:36, 943.07 examples/s]34726 examples [00:36, 955.45 examples/s]34822 examples [00:36, 955.39 examples/s]34918 examples [00:36, 954.41 examples/s]35014 examples [00:37, 951.07 examples/s]35110 examples [00:37, 945.85 examples/s]35205 examples [00:37, 938.57 examples/s]35301 examples [00:37, 941.49 examples/s]35396 examples [00:37, 937.22 examples/s]35496 examples [00:37, 953.63 examples/s]35596 examples [00:37, 964.55 examples/s]35693 examples [00:37, 955.88 examples/s]35790 examples [00:37, 958.74 examples/s]35887 examples [00:37, 961.44 examples/s]35984 examples [00:38, 955.97 examples/s]36084 examples [00:38, 966.92 examples/s]36182 examples [00:38, 969.54 examples/s]36282 examples [00:38, 975.54 examples/s]36380 examples [00:38, 962.49 examples/s]36477 examples [00:38, 948.40 examples/s]36572 examples [00:38, 947.14 examples/s]36672 examples [00:38, 962.07 examples/s]36769 examples [00:38, 960.49 examples/s]36866 examples [00:38, 960.69 examples/s]36963 examples [00:39, 959.87 examples/s]37061 examples [00:39, 964.86 examples/s]37158 examples [00:39, 963.99 examples/s]37255 examples [00:39, 952.21 examples/s]37354 examples [00:39, 960.72 examples/s]37451 examples [00:39, 954.29 examples/s]37547 examples [00:39, 941.68 examples/s]37642 examples [00:39, 924.52 examples/s]37737 examples [00:39, 931.39 examples/s]37836 examples [00:40, 946.27 examples/s]37931 examples [00:40, 935.92 examples/s]38025 examples [00:40, 934.42 examples/s]38119 examples [00:40, 923.13 examples/s]38215 examples [00:40, 933.40 examples/s]38311 examples [00:40, 938.84 examples/s]38405 examples [00:40, 927.37 examples/s]38501 examples [00:40, 936.45 examples/s]38596 examples [00:40, 939.62 examples/s]38693 examples [00:40, 947.95 examples/s]38791 examples [00:41, 956.91 examples/s]38887 examples [00:41, 955.51 examples/s]38983 examples [00:41, 944.10 examples/s]39078 examples [00:41, 945.29 examples/s]39173 examples [00:41, 943.17 examples/s]39271 examples [00:41, 951.30 examples/s]39367 examples [00:41, 943.52 examples/s]39464 examples [00:41, 949.02 examples/s]39559 examples [00:41, 933.60 examples/s]39655 examples [00:41, 939.00 examples/s]39754 examples [00:42, 953.51 examples/s]39850 examples [00:42, 953.07 examples/s]39946 examples [00:42, 946.33 examples/s]40041 examples [00:42, 881.32 examples/s]40141 examples [00:42, 913.81 examples/s]40236 examples [00:42, 923.56 examples/s]40330 examples [00:42, 908.43 examples/s]40423 examples [00:42, 912.31 examples/s]40516 examples [00:42, 916.52 examples/s]40608 examples [00:42, 906.14 examples/s]40702 examples [00:43, 914.49 examples/s]40795 examples [00:43, 918.23 examples/s]40891 examples [00:43, 929.68 examples/s]40987 examples [00:43, 937.29 examples/s]41081 examples [00:43, 925.63 examples/s]41176 examples [00:43, 930.88 examples/s]41270 examples [00:43, 932.15 examples/s]41370 examples [00:43, 949.79 examples/s]41466 examples [00:43, 947.40 examples/s]41567 examples [00:43, 964.21 examples/s]41668 examples [00:44, 975.14 examples/s]41766 examples [00:44, 967.06 examples/s]41868 examples [00:44, 979.94 examples/s]41967 examples [00:44, 975.11 examples/s]42065 examples [00:44, 974.34 examples/s]42163 examples [00:44, 964.00 examples/s]42260 examples [00:44, 940.33 examples/s]42357 examples [00:44, 948.54 examples/s]42454 examples [00:44, 952.23 examples/s]42550 examples [00:45, 952.21 examples/s]42646 examples [00:45, 946.77 examples/s]42741 examples [00:45, 936.07 examples/s]42835 examples [00:45, 936.22 examples/s]42930 examples [00:45, 939.49 examples/s]43024 examples [00:45, 904.70 examples/s]43122 examples [00:45, 923.78 examples/s]43215 examples [00:45, 904.70 examples/s]43315 examples [00:45, 929.61 examples/s]43413 examples [00:45, 942.15 examples/s]43511 examples [00:46, 952.94 examples/s]43610 examples [00:46, 963.19 examples/s]43707 examples [00:46, 962.81 examples/s]43804 examples [00:46, 961.72 examples/s]43901 examples [00:46, 959.86 examples/s]43998 examples [00:46, 962.26 examples/s]44096 examples [00:46, 966.51 examples/s]44193 examples [00:46, 961.62 examples/s]44290 examples [00:46, 954.29 examples/s]44386 examples [00:46, 942.38 examples/s]44485 examples [00:47, 953.89 examples/s]44581 examples [00:47, 940.02 examples/s]44676 examples [00:47, 927.77 examples/s]44774 examples [00:47, 940.60 examples/s]44870 examples [00:47, 944.79 examples/s]44968 examples [00:47, 952.60 examples/s]45064 examples [00:47, 936.53 examples/s]45158 examples [00:47, 933.46 examples/s]45255 examples [00:47, 941.72 examples/s]45350 examples [00:47, 929.02 examples/s]45447 examples [00:48, 938.61 examples/s]45544 examples [00:48, 947.70 examples/s]45643 examples [00:48, 957.21 examples/s]45739 examples [00:48, 953.43 examples/s]45835 examples [00:48, 942.21 examples/s]45930 examples [00:48, 939.84 examples/s]46025 examples [00:48, 930.75 examples/s]46119 examples [00:48, 914.47 examples/s]46221 examples [00:48, 941.09 examples/s]46316 examples [00:49, 933.45 examples/s]46412 examples [00:49, 939.39 examples/s]46508 examples [00:49, 943.27 examples/s]46603 examples [00:49, 938.35 examples/s]46697 examples [00:49, 936.54 examples/s]46795 examples [00:49, 948.74 examples/s]46890 examples [00:49, 938.70 examples/s]46985 examples [00:49, 939.17 examples/s]47079 examples [00:49, 936.71 examples/s]47177 examples [00:49, 948.63 examples/s]47273 examples [00:50, 949.75 examples/s]47369 examples [00:50, 950.23 examples/s]47465 examples [00:50, 948.49 examples/s]47564 examples [00:50, 960.34 examples/s]47661 examples [00:50, 960.45 examples/s]47759 examples [00:50, 963.49 examples/s]47856 examples [00:50, 954.21 examples/s]47952 examples [00:50, 930.67 examples/s]48047 examples [00:50, 934.96 examples/s]48144 examples [00:50, 943.17 examples/s]48243 examples [00:51, 955.33 examples/s]48341 examples [00:51, 962.26 examples/s]48438 examples [00:51, 960.18 examples/s]48537 examples [00:51, 967.40 examples/s]48634 examples [00:51, 962.88 examples/s]48731 examples [00:51, 956.70 examples/s]48827 examples [00:51, 956.11 examples/s]48923 examples [00:51, 950.98 examples/s]49019 examples [00:51, 929.67 examples/s]49113 examples [00:51, 932.13 examples/s]49209 examples [00:52, 937.58 examples/s]49304 examples [00:52, 938.87 examples/s]49398 examples [00:52, 936.83 examples/s]49494 examples [00:52, 942.34 examples/s]49589 examples [00:52, 938.00 examples/s]49684 examples [00:52, 940.76 examples/s]49781 examples [00:52, 947.03 examples/s]49876 examples [00:52, 941.67 examples/s]49976 examples [00:52, 955.93 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 6968/50000 [00:00<00:00, 69673.75 examples/s] 37%|███▋      | 18385/50000 [00:00<00:00, 78898.18 examples/s] 60%|█████▉    | 29853/50000 [00:00<00:00, 87045.61 examples/s] 82%|████████▏ | 40922/50000 [00:00<00:00, 93004.41 examples/s]                                                               0 examples [00:00, ? examples/s]64 examples [00:00, 636.34 examples/s]156 examples [00:00, 700.23 examples/s]250 examples [00:00, 756.83 examples/s]337 examples [00:00, 787.33 examples/s]431 examples [00:00, 827.61 examples/s]524 examples [00:00, 854.03 examples/s]617 examples [00:00, 873.73 examples/s]717 examples [00:00, 906.50 examples/s]808 examples [00:00, 907.03 examples/s]904 examples [00:01, 920.88 examples/s]996 examples [00:01, 918.50 examples/s]1090 examples [00:01, 922.81 examples/s]1188 examples [00:01, 936.93 examples/s]1282 examples [00:01, 936.54 examples/s]1376 examples [00:01, 698.91 examples/s]1468 examples [00:01, 752.01 examples/s]1561 examples [00:01, 796.78 examples/s]1654 examples [00:01, 830.72 examples/s]1750 examples [00:02, 864.68 examples/s]1847 examples [00:02, 893.10 examples/s]1942 examples [00:02, 907.02 examples/s]2039 examples [00:02, 925.04 examples/s]2133 examples [00:02, 914.26 examples/s]2231 examples [00:02, 929.93 examples/s]2329 examples [00:02, 942.27 examples/s]2424 examples [00:02, 940.04 examples/s]2521 examples [00:02, 948.05 examples/s]2617 examples [00:02, 941.75 examples/s]2712 examples [00:03, 943.39 examples/s]2808 examples [00:03, 947.18 examples/s]2907 examples [00:03, 957.79 examples/s]3005 examples [00:03, 962.77 examples/s]3102 examples [00:03, 909.97 examples/s]3198 examples [00:03, 923.94 examples/s]3294 examples [00:03, 933.01 examples/s]3390 examples [00:03, 938.97 examples/s]3486 examples [00:03, 943.37 examples/s]3581 examples [00:03, 927.88 examples/s]3682 examples [00:04, 949.14 examples/s]3781 examples [00:04, 958.82 examples/s]3878 examples [00:04, 960.84 examples/s]3976 examples [00:04, 963.89 examples/s]4074 examples [00:04, 967.96 examples/s]4171 examples [00:04, 949.05 examples/s]4270 examples [00:04, 958.45 examples/s]4368 examples [00:04, 962.87 examples/s]4465 examples [00:04, 961.80 examples/s]4562 examples [00:04, 953.68 examples/s]4658 examples [00:05, 943.55 examples/s]4754 examples [00:05, 946.25 examples/s]4852 examples [00:05, 955.97 examples/s]4948 examples [00:05, 905.95 examples/s]5040 examples [00:05, 898.95 examples/s]5136 examples [00:05, 915.36 examples/s]5231 examples [00:05, 924.30 examples/s]5327 examples [00:05, 933.31 examples/s]5421 examples [00:05, 934.16 examples/s]5515 examples [00:06, 918.46 examples/s]5612 examples [00:06, 931.98 examples/s]5708 examples [00:06, 939.21 examples/s]5803 examples [00:06, 918.76 examples/s]5896 examples [00:06, 920.31 examples/s]5990 examples [00:06, 922.81 examples/s]6086 examples [00:06, 931.79 examples/s]6180 examples [00:06, 926.04 examples/s]6275 examples [00:06, 932.92 examples/s]6369 examples [00:06, 933.99 examples/s]6463 examples [00:07, 928.80 examples/s]6559 examples [00:07, 935.94 examples/s]6655 examples [00:07, 940.87 examples/s]6750 examples [00:07, 940.34 examples/s]6846 examples [00:07, 945.14 examples/s]6941 examples [00:07, 907.43 examples/s]7033 examples [00:07, 898.94 examples/s]7125 examples [00:07, 904.65 examples/s]7220 examples [00:07, 916.18 examples/s]7312 examples [00:07, 896.26 examples/s]7408 examples [00:08, 912.12 examples/s]7500 examples [00:08, 878.91 examples/s]7593 examples [00:08, 893.33 examples/s]7683 examples [00:08, 869.98 examples/s]7771 examples [00:08, 840.02 examples/s]7861 examples [00:08, 855.67 examples/s]7956 examples [00:08, 880.87 examples/s]8052 examples [00:08, 901.38 examples/s]8149 examples [00:08, 919.16 examples/s]8245 examples [00:09, 928.02 examples/s]8344 examples [00:09, 943.15 examples/s]8439 examples [00:09, 942.01 examples/s]8537 examples [00:09, 951.31 examples/s]8633 examples [00:09, 934.82 examples/s]8727 examples [00:09, 930.99 examples/s]8824 examples [00:09, 941.93 examples/s]8919 examples [00:09, 938.54 examples/s]9015 examples [00:09, 943.27 examples/s]9110 examples [00:09, 944.22 examples/s]9205 examples [00:10, 928.41 examples/s]9300 examples [00:10, 934.29 examples/s]9397 examples [00:10, 942.95 examples/s]9492 examples [00:10, 943.40 examples/s]9588 examples [00:10, 947.98 examples/s]9683 examples [00:10, 931.14 examples/s]9777 examples [00:10, 887.15 examples/s]9871 examples [00:10, 901.88 examples/s]9963 examples [00:10, 904.75 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteN06SRM/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteN06SRM/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0052a80950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0052a80950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0052a80950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f009e46cc18>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7effe4cb2860>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/resnet34_benchmark_mnist.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/MNIST\/",
    "dataset":"MNIST",
    "data_type":"tch_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tch_dataset_start",
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"torchvision.datasets:MNIST",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars":false,
          "arg":{

          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0052a80950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0052a80950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0052a80950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f004ffd2320>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f004ffd2080>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/recommender\/",
    "dataset":"IMDB_sample.txt",
    "data_type":"csv_dataset",
    "batch_size":64,
    "train":true
  },
  "preprocessors":[
    {
      "uri":"mlmodels.model_tch.textcnn:split_train_valid",
      "args":{
        "frac":0.99
      }
    },
    {
      "uri":"mlmodels.model_tch.textcnn:create_tabular_dataset",
      "args":{
        "lang":"en",
        "pretrained_emb":"glove.6B.300d"
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7effca590268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7effca590268> 

  function with postional parmater data_info <function split_train_valid at 0x7effca590268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7effca590378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7effca590378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7effca590378> , (data_info, **args) 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:02<75:02:23, 3.19kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:02<52:45:26, 4.54kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:02<36:58:41, 6.48kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<25:52:41, 9.25kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:03<18:03:37, 13.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.22M/862M [00:03<12:33:41, 18.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:03<8:45:22, 26.9kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:03<6:05:47, 38.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.0M/862M [00:03<4:14:51, 54.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.6M/862M [00:03<2:57:31, 78.4kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.6M/862M [00:03<2:03:46, 112kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 35.4M/862M [00:03<1:26:13, 160kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.2M/862M [00:03<1:00:11, 228kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.3M/862M [00:04<41:57, 325kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 47.8M/862M [00:04<29:22, 462kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:04<20:53, 646kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:06<16:28, 815kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:06<15:55, 844kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:06<12:18, 1.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.7M/862M [00:06<08:53, 1.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:08<09:54, 1.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:08<08:36, 1.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:08<06:21, 2.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:10<07:08, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:10<07:48, 1.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:10<06:03, 2.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.6M/862M [00:10<04:24, 3.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:12<08:54, 1.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:12<07:34, 1.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:12<05:37, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:14<07:03, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:14<06:16, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:14<04:42, 2.79MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:16<06:24, 2.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:16<07:08, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:16<05:33, 2.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<04:02, 3.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:18<11:59, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:18<09:42, 1.34MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:18<07:06, 1.83MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:20<08:00, 1.62MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:20<06:54, 1.87MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.1M/862M [00:20<05:09, 2.50MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:22<06:38, 1.94MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:22<05:57, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:22<04:29, 2.86MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:24<06:10, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:24<06:55, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:24<05:29, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:25<04:22, 2.92MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:26<9:21:51, 22.7kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:26<6:33:39, 32.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<4:34:42, 46.2kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<3:24:05, 62.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<2:25:26, 87.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<1:42:20, 124kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<1:13:23, 172kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<52:39, 239kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:30<37:06, 339kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<28:49, 435kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<22:41, 552kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<16:30, 759kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<11:38, 1.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<12:12:20, 17.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<8:33:38, 24.3kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:34<5:59:06, 34.6kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<4:13:34, 48.9kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<2:59:57, 68.9kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<2:06:28, 97.9kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:28:19, 140kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<2:30:03, 82.2kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<1:46:13, 116kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:38<1:14:31, 165kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<54:53, 223kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<39:38, 309kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:40<28:00, 437kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<22:24, 544kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<18:15, 668kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<13:20, 913kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<09:27, 1.28MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:43<35:48, 339kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<26:18, 461kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:44<18:40, 648kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:45<15:52, 760kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<13:32, 890kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<09:59, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:46<07:08, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<11:56, 1.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<09:34, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<06:56, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<07:39, 1.56MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<07:46, 1.53MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<06:01, 1.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:51<06:08, 1.93MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<06:41, 1.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<05:17, 2.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<03:50, 3.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<1:26:59, 136kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<1:02:05, 190kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<43:40, 269kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:55<33:13, 353kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:55<24:13, 483kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<17:09, 681kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<12:10, 957kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:57<1:35:29, 122kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:57<1:09:10, 168kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<48:57, 238kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:59<35:59, 322kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<26:21, 439kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<18:42, 618kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<15:45, 731kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<12:12, 943kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<08:46, 1.31MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<08:49, 1.30MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<08:28, 1.35MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<06:53, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<05:03, 2.26MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<03:49, 2.98MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<03:10, 3.58MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<1:13:08, 156kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<59:05, 193kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<43:23, 262kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<30:46, 369kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:05<21:50, 519kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:06<15:36, 725kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<23:26, 482kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<20:03, 564kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<14:56, 756kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<10:47, 1.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<07:51, 1.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<11:12, 1.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<14:39, 766kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<12:01, 934kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<08:53, 1.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<06:30, 1.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:11<07:53, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:11<08:43, 1.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<06:46, 1.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:11<05:03, 2.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<03:49, 2.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<11:42, 948kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<14:15, 778kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<11:14, 987kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<08:12, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<06:13, 1.78MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<04:34, 2.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<14:28, 762kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<16:06, 685kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<12:45, 864kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:15<09:21, 1.18MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<06:46, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<06:02, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:17<6:32:35, 27.9kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:17<4:36:07, 39.7kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<3:13:22, 56.6kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<2:15:15, 80.7kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:19<1:39:05, 110kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<1:15:14, 145kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<54:03, 201kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:19<38:07, 285kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<26:55, 403kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<23:22, 463kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<22:12, 488kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<16:55, 640kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<12:12, 885kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<08:45, 1.23MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<10:50, 992kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<12:48, 840kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<10:22, 1.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:23<07:38, 1.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<05:35, 1.91MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<08:34, 1.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<11:10, 957kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<09:13, 1.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:25<06:47, 1.57MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<04:59, 2.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<07:54, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<10:45, 987kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<08:53, 1.19MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:27<06:35, 1.61MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<04:49, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<07:56, 1.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<11:14, 938kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<09:12, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:29<06:46, 1.55MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<04:59, 2.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:31<07:44, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:31<11:04, 947kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<08:52, 1.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:31<06:30, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:31<04:56, 2.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<03:41, 2.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:33<20:21, 511kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:33<19:52, 524kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<15:14, 683kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:33<11:01, 942kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<07:55, 1.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<09:55, 1.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<11:57, 865kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<09:43, 1.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:35<07:09, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<05:13, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:36<08:20, 1.23MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<10:50, 948kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<08:56, 1.15MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:37<06:33, 1.56MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<04:48, 2.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<03:42, 2.75MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<21:37, 472kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<20:06, 507kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<15:22, 663kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:39<11:05, 918kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<07:59, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<09:44, 1.04MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<12:16, 825kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:41<09:53, 1.02MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:41<07:14, 1.40MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<05:17, 1.90MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<07:56, 1.27MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<10:27, 963kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:43<08:33, 1.18MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:43<06:17, 1.60MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:43<04:38, 2.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<07:37, 1.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<10:10, 982kB/s] .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<08:20, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:45<06:08, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:30, 2.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<08:12, 1.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<10:33, 941kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<08:23, 1.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:47<06:12, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<04:31, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<03:44, 2.64MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<15:17, 645kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<17:35, 560kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<13:51, 712kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:49<10:07, 971kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:49<07:22, 1.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<05:30, 1.78MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<13:47, 710kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<12:52, 761kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<09:38, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:51<06:58, 1.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:51<05:19, 1.83MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<03:56, 2.46MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<36:04, 270kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<31:13, 311kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<23:19, 417kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:53<16:41, 581kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<11:55, 812kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<11:20, 852kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<13:47, 700kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<11:02, 874kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:55<08:06, 1.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:55<05:52, 1.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<07:22, 1.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<10:25, 919kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<08:37, 1.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<06:20, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:57<04:40, 2.04MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:58<06:43, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:58<09:50, 967kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<08:13, 1.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:59<06:07, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:59<04:30, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<03:21, 2.82MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<42:26, 222kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<34:20, 275kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<25:16, 373kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:01<17:57, 524kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<12:45, 736kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<12:44, 735kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<13:33, 691kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:03<10:43, 874kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:03<07:46, 1.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<05:35, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<04:16, 2.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<36:29, 255kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<30:07, 309kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<22:13, 418kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:05<15:50, 585kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<11:14, 822kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<12:33, 735kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<13:15, 696kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<10:25, 884kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:07<07:32, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<05:27, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<07:26, 1.23MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<09:40, 947kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<07:53, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:09<05:47, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<04:14, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<07:40, 1.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<09:47, 929kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<07:58, 1.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<05:51, 1.55MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:18, 2.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<07:00, 1.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<09:17, 971kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<07:36, 1.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<05:34, 1.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:05, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<07:42, 1.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<09:46, 916kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<07:45, 1.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:15<05:43, 1.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<04:11, 2.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<07:39, 1.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<09:42, 915kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<07:53, 1.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:17<05:50, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<04:16, 2.06MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<07:10, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<09:18, 948kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:18<07:35, 1.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:19<05:35, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:06, 2.13MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<07:34, 1.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<09:34, 914kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<07:45, 1.13MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<05:43, 1.52MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:10, 2.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<07:06, 1.22MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<07:04, 1.23MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<05:27, 1.59MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<04:01, 2.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<05:15, 1.64MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<07:52, 1.09MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<06:33, 1.31MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:25<04:50, 1.77MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<03:34, 2.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:26<07:06, 1.20MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:26<09:06, 937kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:26<07:24, 1.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<05:29, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<04:01, 2.11MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<06:57, 1.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<08:58, 943kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<07:15, 1.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:29<05:21, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<03:55, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:30<07:47, 1.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:30<09:12, 912kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<07:26, 1.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<05:26, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<03:58, 2.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<05:53, 1.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<08:09, 1.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<06:31, 1.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<04:49, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<03:37, 2.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<04:52, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:34<07:04, 1.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:34<05:55, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<04:24, 1.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<03:14, 2.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<08:02, 1.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<09:16, 883kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<07:15, 1.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<05:19, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<03:52, 2.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:38<09:54, 820kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:38<10:31, 771kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:38<08:15, 982kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<06:01, 1.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<04:22, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:40<10:02, 803kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:40<10:16, 783kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:40<08:02, 1.00MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<05:49, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:11, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<08:25, 948kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<09:09, 871kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<08:13, 970kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:42<06:01, 1.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:20, 1.83MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<03:16, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<36:33, 217kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<28:46, 275kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:44<20:55, 378kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<14:49, 532kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<10:28, 749kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<21:49, 359kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<18:13, 430kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:46<13:24, 584kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<09:35, 815kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<06:50, 1.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:48<10:03, 774kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:48<09:44, 799kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:48<07:24, 1.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<05:19, 1.45MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:53, 1.98MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:50<06:51, 1.13MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:50<07:28, 1.03MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:50<05:53, 1.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<04:17, 1.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<04:53, 1.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<06:04, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<04:52, 1.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<03:34, 2.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<04:26, 1.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<05:34, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:54<04:29, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<03:16, 2.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<04:22, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<05:20, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:56<04:14, 1.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:07, 2.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<04:15, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<05:06, 1.46MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:58<04:02, 1.84MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<02:58, 2.49MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [03:00<04:18, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:00<05:00, 1.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:00<05:02, 1.46MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<03:47, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<03:10, 2.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<02:47, 2.62MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<02:23, 3.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<02:10, 3.36MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<05:25, 1.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<06:19, 1.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<05:06, 1.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:02<03:59, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:02<03:19, 2.18MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<02:47, 2.60MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<02:20, 3.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<02:01, 3.56MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<02:03, 3.50MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<07:45, 931kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:04<2:37:08, 46.0kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:04<1:51:09, 65.0kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:04<1:18:11, 92.2kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:04<55:05, 131kB/s]   .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<38:57, 185kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<27:40, 259kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<19:46, 362kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<14:15, 502kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<1:17:42, 92.1kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<56:42, 126kB/s]   .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<40:17, 177kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:06<28:36, 249kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<20:25, 349kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<14:40, 485kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<10:41, 665kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<07:53, 899kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<13:47, 514kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<11:57, 593kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:08<08:59, 787kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:08<06:43, 1.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<05:07, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<04:00, 1.76MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<03:08, 2.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<02:34, 2.72MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<08:02, 872kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<07:45, 904kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:10<06:04, 1.15MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:10<04:40, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<03:41, 1.90MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<02:59, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<02:30, 2.78MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<02:09, 3.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<10:36, 655kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<09:40, 718kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:12<07:17, 953kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:12<05:34, 1.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<04:14, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<03:29, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<02:52, 2.41MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<02:20, 2.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<02:08, 3.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<14:25, 477kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<12:19, 558kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:14<09:13, 745kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<06:51, 999kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<05:12, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<03:59, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<03:12, 2.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<02:41, 2.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:16<06:26, 1.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:16<06:35, 1.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:16<05:11, 1.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<03:58, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<03:12, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<02:37, 2.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<02:13, 3.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<01:52, 3.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:18<1:28:06, 76.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:18<1:03:42, 106kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:18<45:07, 149kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:18<31:52, 211kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<22:40, 296kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<16:11, 413kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<11:40, 573kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<12:04, 552kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<12:06, 551kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<09:15, 720kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:20<06:51, 971kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<05:07, 1.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<03:54, 1.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:03, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<05:45, 1.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<05:43, 1.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:22<04:27, 1.48MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:22<03:23, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<02:39, 2.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<02:05, 3.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<01:56, 3.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:23<08:45, 746kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:24<09:39, 676kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:24<08:34, 762kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:24<06:24, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<04:44, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<03:32, 1.83MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<02:43, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:25<09:11, 703kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:26<09:02, 715kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:26<06:58, 925kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<05:06, 1.26MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<03:45, 1.71MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<02:57, 2.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:27<04:38, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:28<05:28, 1.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<04:25, 1.44MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<03:18, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<02:31, 2.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<01:54, 3.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:29<16:25, 385kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:30<13:32, 467kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:30<09:55, 637kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:30<07:06, 888kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<05:15, 1.19MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<03:49, 1.64MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<10:42, 585kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<10:52, 575kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:32<08:32, 733kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:32<06:13, 1.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<04:30, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<05:05, 1.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<06:57, 889kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<05:42, 1.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:34<04:13, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:05, 1.98MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<04:31, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<05:51, 1.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<04:49, 1.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:36<03:33, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<02:36, 2.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<06:19, 958kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<07:05, 853kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<05:38, 1.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<04:05, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<02:57, 2.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:39<05:50, 1.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:39<06:18, 948kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<04:54, 1.22MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<03:32, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<02:39, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:41<04:23, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:41<05:17, 1.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<04:13, 1.40MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<03:04, 1.91MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<03:30, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<04:28, 1.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<03:37, 1.61MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<02:39, 2.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<03:17, 1.76MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<04:10, 1.38MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<03:19, 1.74MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<02:27, 2.34MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<03:10, 1.80MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<03:51, 1.48MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<03:05, 1.84MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<02:16, 2.50MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:49<03:20, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:49<03:46, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<02:56, 1.91MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:08, 2.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:51<03:20, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:51<04:01, 1.38MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<03:14, 1.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<02:22, 2.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:53<03:18, 1.66MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:53<03:35, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<02:49, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<02:03, 2.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<03:59, 1.36MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<04:11, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:55<03:13, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:19, 2.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:57<03:21, 1.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:57<03:23, 1.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:57<02:38, 2.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<01:54, 2.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:59<06:25, 824kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:59<05:31, 958kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<04:06, 1.28MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:55, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:01<09:04, 576kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:01<07:19, 713kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<05:21, 973kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:46, 1.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:03<15:34, 331kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:03<12:04, 426kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<08:43, 589kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<06:08, 831kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<07:31, 676kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<06:26, 789kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:05<04:47, 1.06MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:24, 1.48MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<06:13, 806kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<05:09, 971kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:07<03:48, 1.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:09<03:30, 1.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:09<04:38, 1.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:09<03:47, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<02:45, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:11<02:56, 1.66MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:11<02:46, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<02:06, 2.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:13<02:21, 2.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:13<02:35, 1.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<02:02, 2.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:15<02:11, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:15<03:27, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:15<02:52, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<02:06, 2.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:17<02:33, 1.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:17<02:42, 1.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:17<02:06, 2.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<01:40, 2.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:19<3:27:42, 22.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:19<2:25:37, 31.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<1:41:25, 45.0kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:21<1:11:34, 63.3kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:21<51:55, 87.3kB/s]  .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:21<36:39, 123kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<25:37, 176kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:23<18:51, 237kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:23<14:03, 317kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:23<10:01, 443kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<07:39, 574kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<07:08, 616kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:25<05:21, 819kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<03:52, 1.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:50, 1.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<03:36, 1.20MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<05:32, 781kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<04:35, 940kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<03:21, 1.28MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:29, 1.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:50, 2.32MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<12:44, 334kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<10:12, 417kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<07:27, 569kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<05:18, 796kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<03:48, 1.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:31<05:14, 800kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:31<05:55, 707kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:31<04:45, 879kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:31<03:28, 1.20MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:31, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<03:09, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<04:26, 927kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<03:39, 1.12MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:33<02:42, 1.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<01:59, 2.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<02:47, 1.45MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<04:08, 978kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<03:21, 1.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<02:29, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<01:50, 2.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:22, 2.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<05:18, 751kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<04:44, 840kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:37<03:33, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:37<02:35, 1.53MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:53, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<09:18, 421kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<08:37, 454kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:39<06:31, 599kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<04:41, 828kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<03:21, 1.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<03:57, 970kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<04:38, 829kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<03:42, 1.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<02:43, 1.40MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:59, 1.91MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:42<02:54, 1.30MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:43<03:51, 977kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:43<03:09, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:43<02:19, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:42, 2.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<02:55, 1.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<03:50, 965kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<03:07, 1.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<02:17, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<01:40, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:17, 2.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<10:04, 361kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<08:48, 413kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<06:36, 550kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<04:43, 764kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<03:21, 1.06MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<04:01, 886kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<04:44, 752kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<03:44, 951kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<02:42, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<01:57, 1.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:50<03:15, 1.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:50<03:58, 879kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:51<03:12, 1.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<02:20, 1.48MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<01:42, 2.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:52<02:52, 1.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:52<03:40, 934kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<02:59, 1.15MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:53<02:11, 1.56MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:53<01:35, 2.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<02:37, 1.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<03:38, 921kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<02:57, 1.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<02:10, 1.53MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<01:35, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<02:34, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<03:25, 961kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<02:48, 1.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<02:03, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<01:30, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<02:35, 1.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<02:33, 1.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<01:57, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<01:25, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<01:06, 2.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:00<02:20, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:00<03:09, 996kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:00<02:35, 1.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<01:55, 1.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:23, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:02<02:28, 1.25MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:02<03:13, 956kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:02<02:37, 1.17MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:03<01:56, 1.58MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<01:24, 2.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:04<02:27, 1.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:04<03:10, 947kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<02:34, 1.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:05<01:53, 1.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:22, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<02:50, 1.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:06<03:16, 897kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<02:38, 1.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<01:55, 1.52MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:23, 2.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:08<02:32, 1.13MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:08<03:11, 902kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:08<02:33, 1.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<01:52, 1.52MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:21, 2.08MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:10<03:00, 935kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:10<03:21, 838kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:10<02:37, 1.07MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<01:54, 1.45MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:22, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<02:29, 1.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<02:56, 929kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<02:21, 1.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<01:43, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:14, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<03:03, 875kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<03:18, 806kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<02:36, 1.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:53, 1.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:21, 1.92MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:16<02:59, 868kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:16<03:08, 826kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<02:25, 1.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:45, 1.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:15, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<03:08, 805kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<03:07, 808kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:18<02:26, 1.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:45, 1.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:15, 1.96MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<04:39, 528kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<04:10, 589kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:20<03:09, 777kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<02:14, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:35, 1.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:22<09:21, 256kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:22<07:26, 322kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:22<05:24, 441kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<03:48, 621kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:24<03:09, 738kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:24<03:04, 756kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:24<02:20, 987kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:40, 1.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<01:40, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<01:57, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<01:33, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:07, 1.98MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:28<01:19, 1.65MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:28<01:38, 1.33MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<01:17, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<00:56, 2.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<01:13, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<01:32, 1.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<01:14, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:30<00:54, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<01:09, 1.78MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<01:23, 1.47MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:32<01:05, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:47, 2.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<01:07, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:34<01:21, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:34<01:03, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:34<00:46, 2.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<01:00, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<01:11, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<00:56, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<00:40, 2.77MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<01:08, 1.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<01:17, 1.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<01:00, 1.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<00:43, 2.47MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:37, 2.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:40<1:19:11, 22.4kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:40<55:28, 31.8kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<38:24, 45.4kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<26:35, 64.1kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<19:00, 89.5kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:42<13:18, 127kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<09:08, 181kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<07:03, 232kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<05:17, 308kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:44<03:45, 430kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:35, 610kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<02:51, 548kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<02:20, 666kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<01:41, 913kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<01:11, 1.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<01:26, 1.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<01:21, 1.10MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<01:00, 1.47MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:43, 2.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<00:55, 1.55MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<00:58, 1.47MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<00:44, 1.89MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:31, 2.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:58, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<01:02, 1.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:48, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:33, 2.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:52, 1.46MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:55, 1.38MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<00:42, 1.79MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:30, 2.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:26, 2.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<1:00:56, 19.9kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<42:51, 28.2kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<29:47, 40.3kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<20:10, 57.5kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<14:30, 78.8kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<10:25, 109kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<07:18, 155kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<04:57, 221kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<03:50, 280kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<02:55, 366kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<02:05, 507kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:25, 717kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<01:28, 680kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<01:11, 836kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:52, 1.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:04<00:44, 1.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:04<00:58, 965kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:04<00:46, 1.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<00:33, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:32, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:30, 1.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:22, 2.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:23, 2.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:35, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:08<00:29, 1.61MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:21, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:24, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:10<00:25, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:10<00:19, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:19, 2.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:21, 1.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<00:15, 2.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:11, 3.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:26, 1.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:25, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<00:18, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:17, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:18, 1.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:16<00:13, 2.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:13, 2.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:19, 1.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:15, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:10, 2.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:12, 1.86MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:12, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:09, 2.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:06, 3.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<00:14, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:13, 1.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:09, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<00:08, 1.82MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:08, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:06, 2.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:03, 3.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:25<00:09, 1.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:08, 1.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<00:05, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:06, 991kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:05, 1.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:03, 1.49MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:02, 1.12MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:02, 1.00MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:30<00:01, 1.28MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:30<00:00, 1.74MB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<181:28:21,  1.63s/it]  0%|          | 760/400000 [00:01<126:47:39,  1.14s/it]  0%|          | 1572/400000 [00:01<88:34:46,  1.25it/s]  1%|          | 2393/400000 [00:01<61:52:54,  1.78it/s]  1%|          | 3124/400000 [00:02<43:14:32,  2.55it/s]  1%|          | 3921/400000 [00:02<30:12:46,  3.64it/s]  1%|          | 4699/400000 [00:02<21:06:42,  5.20it/s]  1%|▏         | 5526/400000 [00:02<14:45:04,  7.43it/s]  2%|▏         | 6321/400000 [00:02<10:18:33, 10.61it/s]  2%|▏         | 7166/400000 [00:02<7:12:17, 15.15it/s]   2%|▏         | 7945/400000 [00:02<5:02:15, 21.62it/s]  2%|▏         | 8733/400000 [00:02<3:31:24, 30.85it/s]  2%|▏         | 9534/400000 [00:02<2:27:55, 43.99it/s]  3%|▎         | 10332/400000 [00:02<1:43:34, 62.70it/s]  3%|▎         | 11179/400000 [00:03<1:12:34, 89.29it/s]  3%|▎         | 11986/400000 [00:03<50:56, 126.94it/s]   3%|▎         | 12786/400000 [00:03<35:50, 180.07it/s]  3%|▎         | 13583/400000 [00:03<25:16, 254.77it/s]  4%|▎         | 14382/400000 [00:03<17:54, 359.05it/s]  4%|▍         | 15197/400000 [00:03<12:44, 503.42it/s]  4%|▍         | 15998/400000 [00:03<09:08, 700.03it/s]  4%|▍         | 16796/400000 [00:03<06:38, 961.84it/s]  4%|▍         | 17608/400000 [00:03<04:52, 1307.65it/s]  5%|▍         | 18441/400000 [00:03<03:38, 1750.24it/s]  5%|▍         | 19246/400000 [00:04<02:46, 2281.50it/s]  5%|▌         | 20059/400000 [00:04<02:10, 2909.34it/s]  5%|▌         | 20862/400000 [00:04<01:45, 3584.98it/s]  5%|▌         | 21659/400000 [00:04<01:28, 4290.79it/s]  6%|▌         | 22476/400000 [00:04<01:15, 5002.57it/s]  6%|▌         | 23305/400000 [00:04<01:06, 5676.99it/s]  6%|▌         | 24115/400000 [00:04<01:00, 6192.80it/s]  6%|▌         | 24918/400000 [00:04<00:57, 6564.47it/s]  6%|▋         | 25709/400000 [00:04<00:54, 6890.84it/s]  7%|▋         | 26536/400000 [00:04<00:51, 7252.41it/s]  7%|▋         | 27380/400000 [00:05<00:49, 7570.06it/s]  7%|▋         | 28193/400000 [00:05<00:48, 7621.54it/s]  7%|▋         | 28999/400000 [00:05<00:47, 7746.12it/s]  7%|▋         | 29802/400000 [00:05<00:48, 7671.99it/s]  8%|▊         | 30600/400000 [00:05<00:47, 7760.04it/s]  8%|▊         | 31403/400000 [00:05<00:47, 7839.09it/s]  8%|▊         | 32222/400000 [00:05<00:46, 7937.28it/s]  8%|▊         | 33023/400000 [00:05<00:47, 7731.19it/s]  8%|▊         | 33803/400000 [00:05<00:47, 7712.17it/s]  9%|▊         | 34627/400000 [00:05<00:46, 7861.51it/s]  9%|▉         | 35438/400000 [00:06<00:45, 7932.98it/s]  9%|▉         | 36281/400000 [00:06<00:45, 8074.60it/s]  9%|▉         | 37092/400000 [00:06<00:45, 8052.38it/s]  9%|▉         | 37900/400000 [00:06<00:46, 7866.34it/s] 10%|▉         | 38689/400000 [00:06<00:46, 7843.64it/s] 10%|▉         | 39495/400000 [00:06<00:45, 7906.82it/s] 10%|█         | 40304/400000 [00:06<00:45, 7957.85it/s] 10%|█         | 41101/400000 [00:06<00:46, 7669.97it/s] 10%|█         | 41908/400000 [00:06<00:46, 7783.80it/s] 11%|█         | 42707/400000 [00:07<00:45, 7841.99it/s] 11%|█         | 43507/400000 [00:07<00:45, 7888.48it/s] 11%|█         | 44298/400000 [00:07<00:45, 7765.18it/s] 11%|█▏        | 45076/400000 [00:07<00:46, 7656.94it/s] 11%|█▏        | 45849/400000 [00:07<00:46, 7677.97it/s] 12%|█▏        | 46618/400000 [00:07<00:46, 7552.88it/s] 12%|█▏        | 47387/400000 [00:07<00:46, 7593.38it/s] 12%|█▏        | 48175/400000 [00:07<00:45, 7676.56it/s] 12%|█▏        | 48944/400000 [00:07<00:45, 7655.90it/s] 12%|█▏        | 49751/400000 [00:07<00:45, 7773.17it/s] 13%|█▎        | 50557/400000 [00:08<00:44, 7855.47it/s] 13%|█▎        | 51344/400000 [00:08<00:44, 7805.18it/s] 13%|█▎        | 52182/400000 [00:08<00:43, 7968.69it/s] 13%|█▎        | 52981/400000 [00:08<00:44, 7848.73it/s] 13%|█▎        | 53773/400000 [00:08<00:44, 7867.68it/s] 14%|█▎        | 54570/400000 [00:08<00:43, 7897.46it/s] 14%|█▍        | 55361/400000 [00:08<00:43, 7850.96it/s] 14%|█▍        | 56151/400000 [00:08<00:43, 7864.89it/s] 14%|█▍        | 56938/400000 [00:08<00:44, 7767.11it/s] 14%|█▍        | 57719/400000 [00:08<00:44, 7776.73it/s] 15%|█▍        | 58511/400000 [00:09<00:43, 7817.64it/s] 15%|█▍        | 59294/400000 [00:09<00:44, 7607.57it/s] 15%|█▌        | 60057/400000 [00:09<00:44, 7596.35it/s] 15%|█▌        | 60818/400000 [00:09<00:44, 7557.26it/s] 15%|█▌        | 61584/400000 [00:09<00:44, 7584.11it/s] 16%|█▌        | 62367/400000 [00:09<00:44, 7654.60it/s] 16%|█▌        | 63146/400000 [00:09<00:43, 7692.83it/s] 16%|█▌        | 63916/400000 [00:09<00:43, 7668.76it/s] 16%|█▌        | 64684/400000 [00:09<00:44, 7552.42it/s] 16%|█▋        | 65458/400000 [00:09<00:43, 7606.91it/s] 17%|█▋        | 66258/400000 [00:10<00:43, 7717.90it/s] 17%|█▋        | 67092/400000 [00:10<00:42, 7892.46it/s] 17%|█▋        | 67941/400000 [00:10<00:41, 8061.78it/s] 17%|█▋        | 68750/400000 [00:10<00:42, 7836.61it/s] 17%|█▋        | 69582/400000 [00:10<00:41, 7973.02it/s] 18%|█▊        | 70397/400000 [00:10<00:41, 8022.90it/s] 18%|█▊        | 71202/400000 [00:10<00:40, 8023.20it/s] 18%|█▊        | 72006/400000 [00:10<00:41, 7970.62it/s] 18%|█▊        | 72805/400000 [00:10<00:42, 7715.46it/s] 18%|█▊        | 73579/400000 [00:10<00:42, 7695.37it/s] 19%|█▊        | 74396/400000 [00:11<00:41, 7830.68it/s] 19%|█▉        | 75202/400000 [00:11<00:41, 7896.95it/s] 19%|█▉        | 76050/400000 [00:11<00:40, 8061.38it/s] 19%|█▉        | 76858/400000 [00:11<00:40, 7945.80it/s] 19%|█▉        | 77655/400000 [00:11<00:41, 7826.58it/s] 20%|█▉        | 78488/400000 [00:11<00:40, 7968.71it/s] 20%|█▉        | 79325/400000 [00:11<00:39, 8083.45it/s] 20%|██        | 80135/400000 [00:11<00:39, 8074.54it/s] 20%|██        | 80944/400000 [00:11<00:39, 8005.26it/s] 20%|██        | 81753/400000 [00:12<00:39, 8028.68it/s] 21%|██        | 82578/400000 [00:12<00:39, 8090.67it/s] 21%|██        | 83388/400000 [00:12<00:39, 8019.91it/s] 21%|██        | 84191/400000 [00:12<00:40, 7813.88it/s] 21%|██        | 84974/400000 [00:12<00:40, 7806.47it/s] 21%|██▏       | 85769/400000 [00:12<00:40, 7847.86it/s] 22%|██▏       | 86555/400000 [00:12<00:40, 7802.07it/s] 22%|██▏       | 87370/400000 [00:12<00:39, 7903.19it/s] 22%|██▏       | 88176/400000 [00:12<00:39, 7949.09it/s] 22%|██▏       | 88972/400000 [00:12<00:39, 7846.32it/s] 22%|██▏       | 89803/400000 [00:13<00:38, 7977.40it/s] 23%|██▎       | 90602/400000 [00:13<00:39, 7804.10it/s] 23%|██▎       | 91417/400000 [00:13<00:39, 7903.52it/s] 23%|██▎       | 92262/400000 [00:13<00:38, 8058.75it/s] 23%|██▎       | 93070/400000 [00:13<00:38, 7985.65it/s] 23%|██▎       | 93890/400000 [00:13<00:38, 8048.37it/s] 24%|██▎       | 94696/400000 [00:13<00:38, 7894.27it/s] 24%|██▍       | 95498/400000 [00:13<00:38, 7930.97it/s] 24%|██▍       | 96293/400000 [00:13<00:38, 7936.56it/s] 24%|██▍       | 97088/400000 [00:13<00:38, 7923.48it/s] 24%|██▍       | 97931/400000 [00:14<00:37, 8067.18it/s] 25%|██▍       | 98739/400000 [00:14<00:37, 8007.27it/s] 25%|██▍       | 99591/400000 [00:14<00:36, 8152.35it/s] 25%|██▌       | 100420/400000 [00:14<00:36, 8190.81it/s] 25%|██▌       | 101240/400000 [00:14<00:37, 7931.85it/s] 26%|██▌       | 102036/400000 [00:14<00:38, 7777.23it/s] 26%|██▌       | 102817/400000 [00:14<00:38, 7678.75it/s] 26%|██▌       | 103587/400000 [00:14<00:39, 7501.01it/s] 26%|██▌       | 104340/400000 [00:14<00:39, 7427.63it/s] 26%|██▋       | 105103/400000 [00:14<00:39, 7487.16it/s] 26%|██▋       | 105867/400000 [00:15<00:39, 7529.94it/s] 27%|██▋       | 106628/400000 [00:15<00:38, 7553.70it/s] 27%|██▋       | 107385/400000 [00:15<00:38, 7523.18it/s] 27%|██▋       | 108145/400000 [00:15<00:38, 7542.84it/s] 27%|██▋       | 108917/400000 [00:15<00:38, 7594.29it/s] 27%|██▋       | 109712/400000 [00:15<00:37, 7695.97it/s] 28%|██▊       | 110497/400000 [00:15<00:37, 7740.24it/s] 28%|██▊       | 111272/400000 [00:15<00:37, 7727.96it/s] 28%|██▊       | 112046/400000 [00:15<00:37, 7659.19it/s] 28%|██▊       | 112813/400000 [00:15<00:37, 7616.17it/s] 28%|██▊       | 113628/400000 [00:16<00:36, 7766.97it/s] 29%|██▊       | 114412/400000 [00:16<00:36, 7788.64it/s] 29%|██▉       | 115192/400000 [00:16<00:36, 7788.89it/s] 29%|██▉       | 115972/400000 [00:16<00:36, 7722.91it/s] 29%|██▉       | 116747/400000 [00:16<00:36, 7729.99it/s] 29%|██▉       | 117559/400000 [00:16<00:36, 7842.07it/s] 30%|██▉       | 118385/400000 [00:16<00:35, 7962.09it/s] 30%|██▉       | 119183/400000 [00:16<00:35, 7917.81it/s] 30%|██▉       | 119976/400000 [00:16<00:35, 7914.45it/s] 30%|███       | 120768/400000 [00:17<00:36, 7740.88it/s] 30%|███       | 121544/400000 [00:17<00:36, 7731.98it/s] 31%|███       | 122342/400000 [00:17<00:35, 7804.03it/s] 31%|███       | 123134/400000 [00:17<00:35, 7836.99it/s] 31%|███       | 123919/400000 [00:17<00:35, 7775.42it/s] 31%|███       | 124698/400000 [00:17<00:36, 7634.67it/s] 31%|███▏      | 125463/400000 [00:17<00:36, 7614.37it/s] 32%|███▏      | 126235/400000 [00:17<00:35, 7645.71it/s] 32%|███▏      | 127050/400000 [00:17<00:35, 7789.82it/s] 32%|███▏      | 127863/400000 [00:17<00:34, 7887.53it/s] 32%|███▏      | 128653/400000 [00:18<00:34, 7822.29it/s] 32%|███▏      | 129469/400000 [00:18<00:34, 7918.25it/s] 33%|███▎      | 130298/400000 [00:18<00:33, 8023.81it/s] 33%|███▎      | 131156/400000 [00:18<00:32, 8180.24it/s] 33%|███▎      | 131976/400000 [00:18<00:33, 8004.55it/s] 33%|███▎      | 132779/400000 [00:18<00:33, 7960.10it/s] 33%|███▎      | 133577/400000 [00:18<00:33, 7959.11it/s] 34%|███▎      | 134382/400000 [00:18<00:33, 7983.94it/s] 34%|███▍      | 135182/400000 [00:18<00:33, 7905.24it/s] 34%|███▍      | 135988/400000 [00:18<00:33, 7949.62it/s] 34%|███▍      | 136784/400000 [00:19<00:33, 7889.65it/s] 34%|███▍      | 137574/400000 [00:19<00:33, 7846.81it/s] 35%|███▍      | 138408/400000 [00:19<00:32, 7987.73it/s] 35%|███▍      | 139216/400000 [00:19<00:32, 8009.33it/s] 35%|███▌      | 140018/400000 [00:19<00:32, 7915.33it/s] 35%|███▌      | 140811/400000 [00:19<00:33, 7759.60it/s] 35%|███▌      | 141589/400000 [00:19<00:33, 7612.56it/s] 36%|███▌      | 142352/400000 [00:19<00:33, 7592.30it/s] 36%|███▌      | 143113/400000 [00:19<00:34, 7526.36it/s] 36%|███▌      | 143867/400000 [00:19<00:34, 7418.20it/s] 36%|███▌      | 144653/400000 [00:20<00:33, 7542.43it/s] 36%|███▋      | 145412/400000 [00:20<00:33, 7554.29it/s] 37%|███▋      | 146194/400000 [00:20<00:33, 7629.77it/s] 37%|███▋      | 147022/400000 [00:20<00:32, 7812.52it/s] 37%|███▋      | 147824/400000 [00:20<00:32, 7870.63it/s] 37%|███▋      | 148613/400000 [00:20<00:32, 7797.24it/s] 37%|███▋      | 149440/400000 [00:20<00:31, 7930.50it/s] 38%|███▊      | 150235/400000 [00:20<00:31, 7862.81it/s] 38%|███▊      | 151023/400000 [00:20<00:31, 7817.99it/s] 38%|███▊      | 151806/400000 [00:20<00:32, 7719.62it/s] 38%|███▊      | 152579/400000 [00:21<00:32, 7708.50it/s] 38%|███▊      | 153399/400000 [00:21<00:31, 7847.80it/s] 39%|███▊      | 154229/400000 [00:21<00:30, 7976.29it/s] 39%|███▉      | 155028/400000 [00:21<00:31, 7798.98it/s] 39%|███▉      | 155810/400000 [00:21<00:31, 7701.83it/s] 39%|███▉      | 156595/400000 [00:21<00:31, 7743.47it/s] 39%|███▉      | 157424/400000 [00:21<00:30, 7896.83it/s] 40%|███▉      | 158233/400000 [00:21<00:30, 7951.57it/s] 40%|███▉      | 159030/400000 [00:21<00:30, 7835.67it/s] 40%|███▉      | 159859/400000 [00:21<00:30, 7965.19it/s] 40%|████      | 160657/400000 [00:22<00:30, 7782.08it/s] 40%|████      | 161438/400000 [00:22<00:31, 7650.74it/s] 41%|████      | 162205/400000 [00:22<00:31, 7593.83it/s] 41%|████      | 162966/400000 [00:22<00:31, 7408.26it/s] 41%|████      | 163729/400000 [00:22<00:31, 7472.57it/s] 41%|████      | 164478/400000 [00:22<00:31, 7395.73it/s] 41%|████▏     | 165301/400000 [00:22<00:30, 7624.57it/s] 42%|████▏     | 166067/400000 [00:22<00:30, 7630.14it/s] 42%|████▏     | 166840/400000 [00:22<00:30, 7658.24it/s] 42%|████▏     | 167637/400000 [00:23<00:30, 7743.28it/s] 42%|████▏     | 168413/400000 [00:23<00:29, 7746.12it/s] 42%|████▏     | 169230/400000 [00:23<00:29, 7867.57it/s] 43%|████▎     | 170054/400000 [00:23<00:28, 7974.93it/s] 43%|████▎     | 170862/400000 [00:23<00:28, 8004.61it/s] 43%|████▎     | 171664/400000 [00:23<00:29, 7751.44it/s] 43%|████▎     | 172442/400000 [00:23<00:29, 7686.34it/s] 43%|████▎     | 173224/400000 [00:23<00:29, 7724.41it/s] 44%|████▎     | 174030/400000 [00:23<00:28, 7819.18it/s] 44%|████▎     | 174847/400000 [00:23<00:28, 7919.56it/s] 44%|████▍     | 175653/400000 [00:24<00:28, 7954.79it/s] 44%|████▍     | 176450/400000 [00:24<00:28, 7724.72it/s] 44%|████▍     | 177225/400000 [00:24<00:29, 7681.57it/s] 44%|████▍     | 177995/400000 [00:24<00:29, 7650.35it/s] 45%|████▍     | 178779/400000 [00:24<00:28, 7703.55it/s] 45%|████▍     | 179551/400000 [00:24<00:29, 7410.26it/s] 45%|████▌     | 180304/400000 [00:24<00:29, 7445.69it/s] 45%|████▌     | 181051/400000 [00:24<00:29, 7306.83it/s] 45%|████▌     | 181808/400000 [00:24<00:29, 7381.68it/s] 46%|████▌     | 182620/400000 [00:24<00:28, 7587.04it/s] 46%|████▌     | 183396/400000 [00:25<00:28, 7638.04it/s] 46%|████▌     | 184165/400000 [00:25<00:28, 7653.52it/s] 46%|████▌     | 184947/400000 [00:25<00:27, 7701.99it/s] 46%|████▋     | 185735/400000 [00:25<00:27, 7753.96it/s] 47%|████▋     | 186541/400000 [00:25<00:27, 7841.90it/s] 47%|████▋     | 187327/400000 [00:25<00:27, 7845.23it/s] 47%|████▋     | 188141/400000 [00:25<00:26, 7928.67it/s] 47%|████▋     | 188935/400000 [00:25<00:27, 7722.71it/s] 47%|████▋     | 189714/400000 [00:25<00:27, 7740.79it/s] 48%|████▊     | 190536/400000 [00:25<00:26, 7877.65it/s] 48%|████▊     | 191350/400000 [00:26<00:26, 7953.49it/s] 48%|████▊     | 192163/400000 [00:26<00:25, 8004.83it/s] 48%|████▊     | 192965/400000 [00:26<00:25, 8003.00it/s] 48%|████▊     | 193807/400000 [00:26<00:25, 8122.17it/s] 49%|████▊     | 194621/400000 [00:26<00:25, 8066.26it/s] 49%|████▉     | 195429/400000 [00:26<00:25, 7895.56it/s] 49%|████▉     | 196257/400000 [00:26<00:25, 8004.99it/s] 49%|████▉     | 197059/400000 [00:26<00:25, 8001.64it/s] 49%|████▉     | 197861/400000 [00:26<00:25, 7876.32it/s] 50%|████▉     | 198673/400000 [00:26<00:25, 7947.38it/s] 50%|████▉     | 199469/400000 [00:27<00:25, 7875.82it/s] 50%|█████     | 200261/400000 [00:27<00:25, 7887.84it/s] 50%|█████     | 201051/400000 [00:27<00:25, 7852.79it/s] 50%|█████     | 201837/400000 [00:27<00:26, 7575.01it/s] 51%|█████     | 202633/400000 [00:27<00:25, 7685.74it/s] 51%|█████     | 203404/400000 [00:27<00:25, 7638.40it/s] 51%|█████     | 204188/400000 [00:27<00:25, 7695.41it/s] 51%|█████▏    | 205028/400000 [00:27<00:24, 7893.18it/s] 51%|█████▏    | 205833/400000 [00:27<00:24, 7938.27it/s] 52%|█████▏    | 206629/400000 [00:28<00:24, 7833.96it/s] 52%|█████▏    | 207414/400000 [00:28<00:24, 7802.95it/s] 52%|█████▏    | 208196/400000 [00:28<00:24, 7672.78it/s] 52%|█████▏    | 209046/400000 [00:28<00:24, 7903.11it/s] 52%|█████▏    | 209839/400000 [00:28<00:24, 7821.85it/s] 53%|█████▎    | 210625/400000 [00:28<00:24, 7831.96it/s] 53%|█████▎    | 211425/400000 [00:28<00:23, 7878.04it/s] 53%|█████▎    | 212233/400000 [00:28<00:23, 7937.06it/s] 53%|█████▎    | 213030/400000 [00:28<00:23, 7946.19it/s] 53%|█████▎    | 213839/400000 [00:28<00:23, 7987.08it/s] 54%|█████▎    | 214639/400000 [00:29<00:23, 7946.16it/s] 54%|█████▍    | 215434/400000 [00:29<00:23, 7889.49it/s] 54%|█████▍    | 216273/400000 [00:29<00:22, 8031.00it/s] 54%|█████▍    | 217096/400000 [00:29<00:22, 8088.32it/s] 54%|█████▍    | 217911/400000 [00:29<00:22, 8106.66it/s] 55%|█████▍    | 218739/400000 [00:29<00:22, 8157.87it/s] 55%|█████▍    | 219568/400000 [00:29<00:22, 8196.91it/s] 55%|█████▌    | 220412/400000 [00:29<00:21, 8265.99it/s] 55%|█████▌    | 221257/400000 [00:29<00:21, 8320.07it/s] 56%|█████▌    | 222115/400000 [00:29<00:21, 8395.01it/s] 56%|█████▌    | 222955/400000 [00:30<00:21, 8128.63it/s] 56%|█████▌    | 223770/400000 [00:30<00:22, 7822.25it/s] 56%|█████▌    | 224577/400000 [00:30<00:22, 7893.84it/s] 56%|█████▋    | 225370/400000 [00:30<00:22, 7861.48it/s] 57%|█████▋    | 226190/400000 [00:30<00:21, 7958.54it/s] 57%|█████▋    | 226988/400000 [00:30<00:21, 7873.95it/s] 57%|█████▋    | 227777/400000 [00:30<00:22, 7603.78it/s] 57%|█████▋    | 228596/400000 [00:30<00:22, 7769.93it/s] 57%|█████▋    | 229376/400000 [00:30<00:21, 7775.39it/s] 58%|█████▊    | 230236/400000 [00:30<00:21, 8005.47it/s] 58%|█████▊    | 231040/400000 [00:31<00:21, 7919.33it/s] 58%|█████▊    | 231835/400000 [00:31<00:21, 7813.68it/s] 58%|█████▊    | 232643/400000 [00:31<00:21, 7891.27it/s] 58%|█████▊    | 233474/400000 [00:31<00:20, 8012.03it/s] 59%|█████▊    | 234277/400000 [00:31<00:20, 7996.34it/s] 59%|█████▉    | 235103/400000 [00:31<00:20, 8072.46it/s] 59%|█████▉    | 235912/400000 [00:31<00:20, 7874.61it/s] 59%|█████▉    | 236702/400000 [00:31<00:20, 7833.97it/s] 59%|█████▉    | 237512/400000 [00:31<00:20, 7910.63it/s] 60%|█████▉    | 238324/400000 [00:32<00:20, 7972.29it/s] 60%|█████▉    | 239123/400000 [00:32<00:20, 7889.22it/s] 60%|█████▉    | 239913/400000 [00:32<00:20, 7842.42it/s] 60%|██████    | 240698/400000 [00:32<00:20, 7777.16it/s] 60%|██████    | 241528/400000 [00:32<00:19, 7925.82it/s] 61%|██████    | 242334/400000 [00:32<00:19, 7962.51it/s] 61%|██████    | 243154/400000 [00:32<00:19, 8031.93it/s] 61%|██████    | 243958/400000 [00:32<00:19, 7936.13it/s] 61%|██████    | 244753/400000 [00:32<00:19, 7837.75it/s] 61%|██████▏   | 245582/400000 [00:32<00:19, 7966.20it/s] 62%|██████▏   | 246423/400000 [00:33<00:18, 8094.11it/s] 62%|██████▏   | 247250/400000 [00:33<00:18, 8146.07it/s] 62%|██████▏   | 248066/400000 [00:33<00:19, 7917.26it/s] 62%|██████▏   | 248867/400000 [00:33<00:19, 7944.69it/s] 62%|██████▏   | 249685/400000 [00:33<00:18, 8012.56it/s] 63%|██████▎   | 250505/400000 [00:33<00:18, 8065.94it/s] 63%|██████▎   | 251313/400000 [00:33<00:18, 8069.40it/s] 63%|██████▎   | 252121/400000 [00:33<00:18, 7847.29it/s] 63%|██████▎   | 252917/400000 [00:33<00:18, 7880.49it/s] 63%|██████▎   | 253729/400000 [00:33<00:18, 7949.87it/s] 64%|██████▎   | 254559/400000 [00:34<00:18, 8050.72it/s] 64%|██████▍   | 255366/400000 [00:34<00:18, 8029.67it/s] 64%|██████▍   | 256170/400000 [00:34<00:18, 7986.00it/s] 64%|██████▍   | 256970/400000 [00:34<00:17, 7968.06it/s] 64%|██████▍   | 257768/400000 [00:34<00:17, 7903.27it/s] 65%|██████▍   | 258586/400000 [00:34<00:17, 7984.29it/s] 65%|██████▍   | 259397/400000 [00:34<00:17, 8021.05it/s] 65%|██████▌   | 260200/400000 [00:34<00:17, 7934.45it/s] 65%|██████▌   | 261002/400000 [00:34<00:17, 7959.53it/s] 65%|██████▌   | 261799/400000 [00:34<00:17, 7910.02it/s] 66%|██████▌   | 262639/400000 [00:35<00:17, 8050.52it/s] 66%|██████▌   | 263459/400000 [00:35<00:16, 8091.93it/s] 66%|██████▌   | 264269/400000 [00:35<00:17, 7887.03it/s] 66%|██████▋   | 265060/400000 [00:35<00:17, 7772.91it/s] 66%|██████▋   | 265840/400000 [00:35<00:17, 7779.03it/s] 67%|██████▋   | 266682/400000 [00:35<00:16, 7958.82it/s] 67%|██████▋   | 267535/400000 [00:35<00:16, 8121.04it/s] 67%|██████▋   | 268350/400000 [00:35<00:16, 7991.91it/s] 67%|██████▋   | 269167/400000 [00:35<00:16, 8044.33it/s] 68%|██████▊   | 270011/400000 [00:35<00:15, 8156.83it/s] 68%|██████▊   | 270829/400000 [00:36<00:15, 8149.03it/s] 68%|██████▊   | 271645/400000 [00:36<00:15, 8137.98it/s] 68%|██████▊   | 272460/400000 [00:36<00:15, 8113.29it/s] 68%|██████▊   | 273272/400000 [00:36<00:15, 8032.65it/s] 69%|██████▊   | 274077/400000 [00:36<00:15, 8037.02it/s] 69%|██████▊   | 274882/400000 [00:36<00:15, 7981.27it/s] 69%|██████▉   | 275714/400000 [00:36<00:15, 8078.70it/s] 69%|██████▉   | 276523/400000 [00:36<00:15, 7845.51it/s] 69%|██████▉   | 277310/400000 [00:36<00:15, 7737.85it/s] 70%|██████▉   | 278125/400000 [00:36<00:15, 7854.07it/s] 70%|██████▉   | 278912/400000 [00:37<00:15, 7797.49it/s] 70%|██████▉   | 279693/400000 [00:37<00:15, 7601.11it/s] 70%|███████   | 280476/400000 [00:37<00:15, 7667.90it/s] 70%|███████   | 281256/400000 [00:37<00:15, 7706.37it/s] 71%|███████   | 282062/400000 [00:37<00:15, 7808.90it/s] 71%|███████   | 282844/400000 [00:37<00:14, 7811.20it/s] 71%|███████   | 283634/400000 [00:37<00:14, 7836.34it/s] 71%|███████   | 284419/400000 [00:37<00:14, 7803.35it/s] 71%|███████▏  | 285200/400000 [00:37<00:14, 7723.62it/s] 71%|███████▏  | 285973/400000 [00:38<00:14, 7720.38it/s] 72%|███████▏  | 286805/400000 [00:38<00:14, 7889.04it/s] 72%|███████▏  | 287596/400000 [00:38<00:14, 7825.56it/s] 72%|███████▏  | 288383/400000 [00:38<00:14, 7836.84it/s] 72%|███████▏  | 289185/400000 [00:38<00:14, 7890.24it/s] 72%|███████▏  | 289975/400000 [00:38<00:14, 7822.13it/s] 73%|███████▎  | 290758/400000 [00:38<00:14, 7768.49it/s] 73%|███████▎  | 291536/400000 [00:38<00:14, 7359.39it/s] 73%|███████▎  | 292277/400000 [00:38<00:14, 7284.02it/s] 73%|███████▎  | 293114/400000 [00:38<00:14, 7577.33it/s] 73%|███████▎  | 293907/400000 [00:39<00:13, 7679.37it/s] 74%|███████▎  | 294753/400000 [00:39<00:13, 7896.42it/s] 74%|███████▍  | 295559/400000 [00:39<00:13, 7942.39it/s] 74%|███████▍  | 296357/400000 [00:39<00:13, 7883.14it/s] 74%|███████▍  | 297207/400000 [00:39<00:12, 8058.38it/s] 75%|███████▍  | 298047/400000 [00:39<00:12, 8155.19it/s] 75%|███████▍  | 298885/400000 [00:39<00:12, 8220.70it/s] 75%|███████▍  | 299709/400000 [00:39<00:12, 8183.59it/s] 75%|███████▌  | 300529/400000 [00:39<00:12, 8015.07it/s] 75%|███████▌  | 301381/400000 [00:39<00:12, 8159.04it/s] 76%|███████▌  | 302199/400000 [00:40<00:12, 7965.16it/s] 76%|███████▌  | 303005/400000 [00:40<00:12, 7991.44it/s] 76%|███████▌  | 303847/400000 [00:40<00:11, 8114.31it/s] 76%|███████▌  | 304660/400000 [00:40<00:11, 7968.39it/s] 76%|███████▋  | 305459/400000 [00:40<00:12, 7876.97it/s] 77%|███████▋  | 306249/400000 [00:40<00:11, 7826.50it/s] 77%|███████▋  | 307033/400000 [00:40<00:11, 7791.67it/s] 77%|███████▋  | 307813/400000 [00:40<00:11, 7704.24it/s] 77%|███████▋  | 308585/400000 [00:40<00:12, 7562.78it/s] 77%|███████▋  | 309399/400000 [00:40<00:11, 7726.48it/s] 78%|███████▊  | 310201/400000 [00:41<00:11, 7810.21it/s] 78%|███████▊  | 311042/400000 [00:41<00:11, 7978.95it/s] 78%|███████▊  | 311859/400000 [00:41<00:10, 8033.99it/s] 78%|███████▊  | 312664/400000 [00:41<00:10, 7980.99it/s] 78%|███████▊  | 313464/400000 [00:41<00:10, 7952.15it/s] 79%|███████▊  | 314260/400000 [00:41<00:10, 7915.74it/s] 79%|███████▉  | 315103/400000 [00:41<00:10, 8063.03it/s] 79%|███████▉  | 315911/400000 [00:41<00:10, 7947.76it/s] 79%|███████▉  | 316707/400000 [00:41<00:10, 7888.65it/s] 79%|███████▉  | 317511/400000 [00:42<00:10, 7930.98it/s] 80%|███████▉  | 318305/400000 [00:42<00:10, 7873.03it/s] 80%|███████▉  | 319149/400000 [00:42<00:10, 8034.48it/s] 80%|███████▉  | 319995/400000 [00:42<00:09, 8156.03it/s] 80%|████████  | 320812/400000 [00:42<00:09, 8121.24it/s] 80%|████████  | 321631/400000 [00:42<00:09, 8139.69it/s] 81%|████████  | 322477/400000 [00:42<00:09, 8232.33it/s] 81%|████████  | 323301/400000 [00:42<00:09, 7963.45it/s] 81%|████████  | 324100/400000 [00:42<00:09, 7852.52it/s] 81%|████████  | 324893/400000 [00:42<00:09, 7874.97it/s] 81%|████████▏ | 325682/400000 [00:43<00:09, 7831.86it/s] 82%|████████▏ | 326495/400000 [00:43<00:09, 7918.05it/s] 82%|████████▏ | 327299/400000 [00:43<00:09, 7953.46it/s] 82%|████████▏ | 328118/400000 [00:43<00:08, 8020.28it/s] 82%|████████▏ | 328924/400000 [00:43<00:08, 8031.61it/s] 82%|████████▏ | 329737/400000 [00:43<00:08, 8058.88it/s] 83%|████████▎ | 330550/400000 [00:43<00:08, 8076.49it/s] 83%|████████▎ | 331358/400000 [00:43<00:08, 7950.82it/s] 83%|████████▎ | 332154/400000 [00:43<00:08, 7870.38it/s] 83%|████████▎ | 332973/400000 [00:43<00:08, 7963.07it/s] 83%|████████▎ | 333812/400000 [00:44<00:08, 8084.83it/s] 84%|████████▎ | 334622/400000 [00:44<00:08, 8002.67it/s] 84%|████████▍ | 335443/400000 [00:44<00:08, 8062.11it/s] 84%|████████▍ | 336265/400000 [00:44<00:07, 8105.23it/s] 84%|████████▍ | 337077/400000 [00:44<00:07, 8041.65it/s] 84%|████████▍ | 337915/400000 [00:44<00:07, 8139.14it/s] 85%|████████▍ | 338730/400000 [00:44<00:07, 8124.45it/s] 85%|████████▍ | 339543/400000 [00:44<00:07, 8072.68it/s] 85%|████████▌ | 340351/400000 [00:44<00:07, 8021.05it/s] 85%|████████▌ | 341154/400000 [00:44<00:07, 7967.56it/s] 85%|████████▌ | 341980/400000 [00:45<00:07, 8052.89it/s] 86%|████████▌ | 342786/400000 [00:45<00:07, 7875.38it/s] 86%|████████▌ | 343598/400000 [00:45<00:07, 7946.87it/s] 86%|████████▌ | 344394/400000 [00:45<00:07, 7914.74it/s] 86%|████████▋ | 345187/400000 [00:45<00:06, 7901.48it/s] 86%|████████▋ | 345978/400000 [00:45<00:06, 7883.70it/s] 87%|████████▋ | 346767/400000 [00:45<00:06, 7804.16it/s] 87%|████████▋ | 347577/400000 [00:45<00:06, 7889.73it/s] 87%|████████▋ | 348367/400000 [00:45<00:06, 7844.55it/s] 87%|████████▋ | 349152/400000 [00:45<00:06, 7774.79it/s] 87%|████████▋ | 349996/400000 [00:46<00:06, 7962.20it/s] 88%|████████▊ | 350833/400000 [00:46<00:06, 8080.27it/s] 88%|████████▊ | 351643/400000 [00:46<00:06, 7996.86it/s] 88%|████████▊ | 352473/400000 [00:46<00:05, 8083.97it/s] 88%|████████▊ | 353283/400000 [00:46<00:05, 7973.66it/s] 89%|████████▊ | 354117/400000 [00:46<00:05, 8078.83it/s] 89%|████████▊ | 354963/400000 [00:46<00:05, 8189.02it/s] 89%|████████▉ | 355784/400000 [00:46<00:05, 8185.58it/s] 89%|████████▉ | 356604/400000 [00:46<00:05, 8119.35it/s] 89%|████████▉ | 357417/400000 [00:46<00:05, 8063.11it/s] 90%|████████▉ | 358227/400000 [00:47<00:05, 8072.48it/s] 90%|████████▉ | 359037/400000 [00:47<00:05, 8078.18it/s] 90%|████████▉ | 359846/400000 [00:47<00:05, 7956.76it/s] 90%|█████████ | 360656/400000 [00:47<00:04, 7996.58it/s] 90%|█████████ | 361457/400000 [00:47<00:04, 7978.10it/s] 91%|█████████ | 362316/400000 [00:47<00:04, 8149.26it/s] 91%|█████████ | 363177/400000 [00:47<00:04, 8281.54it/s] 91%|█████████ | 364007/400000 [00:47<00:04, 8242.01it/s] 91%|█████████ | 364833/400000 [00:47<00:04, 8061.28it/s] 91%|█████████▏| 365641/400000 [00:48<00:04, 7945.38it/s] 92%|█████████▏| 366438/400000 [00:48<00:04, 7590.88it/s] 92%|█████████▏| 367231/400000 [00:48<00:04, 7688.22it/s] 92%|█████████▏| 368004/400000 [00:48<00:04, 7688.77it/s] 92%|█████████▏| 368830/400000 [00:48<00:03, 7848.92it/s] 92%|█████████▏| 369621/400000 [00:48<00:03, 7866.08it/s] 93%|█████████▎| 370448/400000 [00:48<00:03, 7980.88it/s] 93%|█████████▎| 371298/400000 [00:48<00:03, 8127.85it/s] 93%|█████████▎| 372113/400000 [00:48<00:03, 8124.11it/s] 93%|█████████▎| 372927/400000 [00:48<00:03, 8055.07it/s] 93%|█████████▎| 373734/400000 [00:49<00:03, 7914.89it/s] 94%|█████████▎| 374551/400000 [00:49<00:03, 7988.12it/s] 94%|█████████▍| 375380/400000 [00:49<00:03, 8071.30it/s] 94%|█████████▍| 376237/400000 [00:49<00:02, 8212.05it/s] 94%|█████████▍| 377060/400000 [00:49<00:02, 8053.33it/s] 94%|█████████▍| 377867/400000 [00:49<00:02, 7920.60it/s] 95%|█████████▍| 378668/400000 [00:49<00:02, 7945.90it/s] 95%|█████████▍| 379479/400000 [00:49<00:02, 7991.54it/s] 95%|█████████▌| 380280/400000 [00:49<00:02, 7963.40it/s] 95%|█████████▌| 381095/400000 [00:49<00:02, 8015.78it/s] 95%|█████████▌| 381942/400000 [00:50<00:02, 8144.96it/s] 96%|█████████▌| 382758/400000 [00:50<00:02, 8050.18it/s] 96%|█████████▌| 383577/400000 [00:50<00:02, 8090.11it/s] 96%|█████████▌| 384396/400000 [00:50<00:01, 8119.44it/s] 96%|█████████▋| 385218/400000 [00:50<00:01, 8146.81it/s] 97%|█████████▋| 386034/400000 [00:50<00:01, 8061.96it/s] 97%|█████████▋| 386841/400000 [00:50<00:01, 8048.94it/s] 97%|█████████▋| 387647/400000 [00:50<00:01, 7856.45it/s] 97%|█████████▋| 388434/400000 [00:50<00:01, 7806.96it/s] 97%|█████████▋| 389216/400000 [00:50<00:01, 7690.99it/s] 97%|█████████▋| 389987/400000 [00:51<00:01, 7639.30it/s] 98%|█████████▊| 390757/400000 [00:51<00:01, 7655.31it/s] 98%|█████████▊| 391552/400000 [00:51<00:01, 7738.91it/s] 98%|█████████▊| 392333/400000 [00:51<00:00, 7759.97it/s] 98%|█████████▊| 393148/400000 [00:51<00:00, 7871.66it/s] 98%|█████████▊| 393947/400000 [00:51<00:00, 7904.25it/s] 99%|█████████▊| 394807/400000 [00:51<00:00, 8100.48it/s] 99%|█████████▉| 395653/400000 [00:51<00:00, 8202.25it/s] 99%|█████████▉| 396486/400000 [00:51<00:00, 8239.70it/s] 99%|█████████▉| 397313/400000 [00:51<00:00, 8248.58it/s]100%|█████████▉| 398139/400000 [00:52<00:00, 8132.87it/s]100%|█████████▉| 398954/400000 [00:52<00:00, 8019.43it/s]100%|█████████▉| 399757/400000 [00:52<00:00, 7887.04it/s]100%|█████████▉| 399999/400000 [00:52<00:00, 7645.27it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7effca28e7b8>, <torchtext.data.dataset.TabularDataset object at 0x7effca28e908>, <torchtext.vocab.Vocab object at 0x7effca28e828>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{

  },
  "preprocessors":[
    {
      "name":"DataReader",
      "uri":"mlmodels.model_tch.util_transformer:TransformerDataReader",
      "args":{
        "train":{
          "uri":"sentence_transformers.readers:NLIDataReader",
          "dataset":"dataset\/text\/AllNLI\/train"
        },
        "test":{
          "uri":"sentence_transformers.readers:STSBenchmarkDataReader",
          "dataset":"dataset\/text\/stsbenchmark\/val"
        }
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.util_transformer:TransformerDataReader {'train': {'uri': 'sentence_transformers.readers:NLIDataReader', 'dataset': 'dataset/text/AllNLI/train'}, 'test': {'uri': 'sentence_transformers.readers:STSBenchmarkDataReader', 'dataset': 'dataset/text/stsbenchmark/val'}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.model_tch.util_transformer.TransformerDataReader'> 
cls_name : TransformerDataReader

  
 Object Creation 

  
 Object Compute 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.json Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 672, in load_function_uri
    return  getattr(importlib.import_module(package), name)
AttributeError: module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 275, in compute
    obj_preprocessor.compute(input_tmp)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/util_transformer.py", line 275, in compute
    test_func = load_function(self.test_reader)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py --do test  

  #### Test unit Dataloader/Dataset   #################################### 

  


 #################### tf_dataset_download 

  tf_dataset_download mlmodels/preprocess/generic:tf_dataset_download {'train_samples': 500, 'test_samples': 500} 

  MNIST 

  Dataset Name is :  mnist 
WARNING:absl:Dataset mnist is hosted on GCS. It will automatically be downloaded to your
local data directory. If you'd instead prefer to read directly from our public
GCS bucket (recommended if you're running on GCP), you can instead set
data_dir=gs://tfds-data/datasets.

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 25.14 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 45.79 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 18.08 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 18.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.90 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.90 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.57 file/s]2020-06-14 15:38:36.472109: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-14 15:38:36.477135: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-14 15:38:36.477416: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5619c8c2d9e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-14 15:38:36.477433: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist_dataset_small.npy', 'test', 'fashion-mnist_small.npy', 'mnist2'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 154654.12it/s] 82%|████████▏ | 8142848/9912422 [00:00<00:08, 220752.91it/s]9920512it [00:00, 45335594.32it/s]                           
0it [00:00, ?it/s]32768it [00:00, 629541.37it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:12, 129942.32it/s]1654784it [00:00, 9856256.13it/s]                          
0it [00:00, ?it/s]8192it [00:00, 187596.16it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw
Processing...
Done!

  


 #################### PandasDataset 

  PandasDataset mlmodels/preprocess/generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': 'ISO-8859-1', 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX'], 'encoding': 'ISO-8859-1'}} 

  


 #################### NumpyDataset 

  NumpyDataset mlmodels/preprocess/generic:NumpyDataset {'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'numpy_loader_args': {}} 

Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/train/mnist.npz
