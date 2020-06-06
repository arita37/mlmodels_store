
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'a4f6eb9a7161522868de1f62953382979dca62d3', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a4f6eb9a7161522868de1f62953382979dca62d3

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/a4f6eb9a7161522868de1f62953382979dca62d3

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f2485ee86a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f2485ee86a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f24f14b00d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f24f14b00d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f250b1ddea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f250b1ddea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f249ed2b950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f249ed2b950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f249ed2b950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 53%|█████▎    | 5218304/9912422 [00:00<00:00, 52181721.24it/s]9920512it [00:00, 34245774.31it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 209352.31it/s]           
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 148043.61it/s]1654784it [00:00, 10513135.84it/s]                         
0it [00:00, ?it/s]8192it [00:00, 154101.66it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f249c38db00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f249c382d68>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f249ed2b598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f249ed2b598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f249ed2b598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:04,  2.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:04,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:03,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:03,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:02,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:02,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:02,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:44,  3.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:44,  3.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:43,  3.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:43,  3.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:43,  3.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:42,  3.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:42,  3.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:42,  3.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:42,  3.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:29,  4.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:29,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:29,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:29,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:29,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:29,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:28,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:28,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:28,  4.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:20,  6.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:20,  6.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:20,  6.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:20,  6.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:19,  6.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:19,  6.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:19,  6.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:19,  6.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:19,  6.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:13,  9.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:13,  9.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:13,  9.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:13,  9.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:13,  9.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:13,  9.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:13,  9.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:13,  9.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:13,  9.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 12.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 12.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:09, 12.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:09, 12.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:09, 12.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:09, 12.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:09, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:09, 12.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 16.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 16.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 21.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 21.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 21.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 21.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 21.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 21.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 21.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 21.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 21.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 27.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 27.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 27.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 27.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 27.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 27.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 27.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 27.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 27.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 34.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 34.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 34.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 34.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 34.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 34.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 34.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 34.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 39.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 39.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 39.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 39.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 39.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 39.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 39.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 46.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 46.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 46.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 46.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 46.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 46.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 46.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 52.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 52.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 52.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 52.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 52.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 52.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 52.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 52.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 52.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 57.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 57.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 57.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 57.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 57.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 57.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 57.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 57.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 61.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 61.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 61.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 61.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 61.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 61.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 61.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 61.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 61.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 64.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 65.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 65.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 65.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 65.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 65.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 65.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 65.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 65.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 65.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 66.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 65.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 65.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 65.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 65.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 65.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 65.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 65.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 65.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 66.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 66.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.69s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.69s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 66.65 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.69s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 66.65 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.91s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.69s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 66.65 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.91s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.91s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.96 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.92s/ url]
0 examples [00:00, ? examples/s]2020-06-06 18:09:55.525745: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-06 18:09:55.539415: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-06 18:09:55.539598: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d7b37c4800 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-06 18:09:55.539616: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
14 examples [00:00, 138.81 examples/s]103 examples [00:00, 185.78 examples/s]193 examples [00:00, 243.65 examples/s]283 examples [00:00, 311.57 examples/s]376 examples [00:00, 388.87 examples/s]465 examples [00:00, 467.40 examples/s]551 examples [00:00, 540.91 examples/s]641 examples [00:00, 613.91 examples/s]731 examples [00:00, 677.96 examples/s]819 examples [00:01, 727.59 examples/s]910 examples [00:01, 772.35 examples/s]1000 examples [00:01, 805.53 examples/s]1090 examples [00:01, 829.38 examples/s]1184 examples [00:01, 858.14 examples/s]1274 examples [00:01, 849.75 examples/s]1362 examples [00:01, 828.93 examples/s]1447 examples [00:01, 824.22 examples/s]1534 examples [00:01, 834.45 examples/s]1620 examples [00:01, 839.26 examples/s]1705 examples [00:02, 829.92 examples/s]1794 examples [00:02, 847.03 examples/s]1884 examples [00:02, 859.43 examples/s]1974 examples [00:02, 870.78 examples/s]2062 examples [00:02, 862.33 examples/s]2149 examples [00:02, 858.78 examples/s]2236 examples [00:02, 847.11 examples/s]2326 examples [00:02, 860.62 examples/s]2413 examples [00:02, 845.76 examples/s]2500 examples [00:02, 851.25 examples/s]2591 examples [00:03, 867.30 examples/s]2678 examples [00:03, 851.83 examples/s]2764 examples [00:03, 847.07 examples/s]2856 examples [00:03, 866.24 examples/s]2946 examples [00:03, 874.02 examples/s]3039 examples [00:03, 889.21 examples/s]3129 examples [00:03, 890.47 examples/s]3220 examples [00:03, 896.17 examples/s]3314 examples [00:03, 908.62 examples/s]3406 examples [00:03, 911.73 examples/s]3498 examples [00:04, 904.49 examples/s]3589 examples [00:04, 886.19 examples/s]3678 examples [00:04, 885.89 examples/s]3767 examples [00:04, 881.52 examples/s]3858 examples [00:04, 888.12 examples/s]3947 examples [00:04, 879.87 examples/s]4036 examples [00:04, 864.71 examples/s]4123 examples [00:04, 847.11 examples/s]4208 examples [00:04, 829.32 examples/s]4293 examples [00:05, 834.26 examples/s]4386 examples [00:05, 859.29 examples/s]4477 examples [00:05, 871.38 examples/s]4567 examples [00:05, 877.62 examples/s]4655 examples [00:05, 878.26 examples/s]4748 examples [00:05, 892.93 examples/s]4838 examples [00:05, 894.65 examples/s]4928 examples [00:05, 873.49 examples/s]5016 examples [00:05, 864.38 examples/s]5103 examples [00:05, 848.07 examples/s]5188 examples [00:06, 848.20 examples/s]5276 examples [00:06, 855.58 examples/s]5363 examples [00:06, 856.74 examples/s]5449 examples [00:06, 846.23 examples/s]5542 examples [00:06, 868.52 examples/s]5634 examples [00:06, 881.54 examples/s]5728 examples [00:06, 896.50 examples/s]5824 examples [00:06, 912.82 examples/s]5916 examples [00:06, 912.28 examples/s]6010 examples [00:06, 920.34 examples/s]6104 examples [00:07, 926.04 examples/s]6201 examples [00:07, 936.73 examples/s]6295 examples [00:07, 934.63 examples/s]6389 examples [00:07, 925.33 examples/s]6482 examples [00:07, 897.02 examples/s]6572 examples [00:07, 891.97 examples/s]6662 examples [00:07, 884.13 examples/s]6751 examples [00:07, 881.84 examples/s]6840 examples [00:07, 866.86 examples/s]6928 examples [00:07, 868.44 examples/s]7015 examples [00:08, 861.34 examples/s]7102 examples [00:08, 838.30 examples/s]7187 examples [00:08, 828.06 examples/s]7271 examples [00:08, 830.83 examples/s]7357 examples [00:08, 838.10 examples/s]7446 examples [00:08, 852.32 examples/s]7532 examples [00:08, 854.36 examples/s]7619 examples [00:08, 857.68 examples/s]7705 examples [00:08, 849.84 examples/s]7791 examples [00:09, 836.46 examples/s]7875 examples [00:09, 804.08 examples/s]7956 examples [00:09, 795.73 examples/s]8038 examples [00:09, 795.66 examples/s]8124 examples [00:09, 812.28 examples/s]8210 examples [00:09, 825.72 examples/s]8295 examples [00:09, 830.98 examples/s]8380 examples [00:09, 835.31 examples/s]8472 examples [00:09, 857.94 examples/s]8561 examples [00:09, 866.84 examples/s]8649 examples [00:10, 868.98 examples/s]8738 examples [00:10, 873.19 examples/s]8828 examples [00:10, 879.17 examples/s]8917 examples [00:10, 881.97 examples/s]9006 examples [00:10, 864.23 examples/s]9095 examples [00:10, 870.21 examples/s]9183 examples [00:10, 869.88 examples/s]9271 examples [00:10, 865.93 examples/s]9358 examples [00:10, 850.24 examples/s]9446 examples [00:10, 857.52 examples/s]9532 examples [00:11, 847.85 examples/s]9617 examples [00:11, 841.66 examples/s]9702 examples [00:11, 841.00 examples/s]9787 examples [00:11, 839.76 examples/s]9874 examples [00:11, 847.02 examples/s]9959 examples [00:11, 837.78 examples/s]10043 examples [00:11, 762.48 examples/s]10121 examples [00:11, 761.74 examples/s]10203 examples [00:11, 776.21 examples/s]10292 examples [00:12, 806.38 examples/s]10380 examples [00:12, 826.67 examples/s]10468 examples [00:12, 840.12 examples/s]10553 examples [00:12, 817.46 examples/s]10636 examples [00:12, 815.24 examples/s]10724 examples [00:12, 833.53 examples/s]10808 examples [00:12, 832.71 examples/s]10898 examples [00:12, 851.47 examples/s]10986 examples [00:12, 858.44 examples/s]11076 examples [00:12, 867.93 examples/s]11171 examples [00:13, 888.06 examples/s]11263 examples [00:13, 895.04 examples/s]11358 examples [00:13, 909.02 examples/s]11450 examples [00:13, 909.97 examples/s]11542 examples [00:13, 910.23 examples/s]11634 examples [00:13, 896.74 examples/s]11724 examples [00:13, 880.40 examples/s]11813 examples [00:13, 866.19 examples/s]11900 examples [00:13, 856.92 examples/s]11986 examples [00:13, 844.20 examples/s]12071 examples [00:14, 843.60 examples/s]12158 examples [00:14, 850.39 examples/s]12244 examples [00:14, 848.65 examples/s]12329 examples [00:14, 826.74 examples/s]12413 examples [00:14, 828.70 examples/s]12500 examples [00:14, 839.79 examples/s]12588 examples [00:14, 849.85 examples/s]12679 examples [00:14, 864.95 examples/s]12766 examples [00:14, 842.27 examples/s]12851 examples [00:14, 843.61 examples/s]12936 examples [00:15, 825.85 examples/s]13019 examples [00:15, 814.16 examples/s]13101 examples [00:15, 808.74 examples/s]13183 examples [00:15, 804.59 examples/s]13267 examples [00:15, 814.87 examples/s]13349 examples [00:15, 806.75 examples/s]13441 examples [00:15, 836.87 examples/s]13532 examples [00:15, 855.42 examples/s]13619 examples [00:15, 859.44 examples/s]13712 examples [00:16, 877.39 examples/s]13805 examples [00:16, 890.85 examples/s]13898 examples [00:16, 900.45 examples/s]13992 examples [00:16, 910.01 examples/s]14085 examples [00:16, 913.75 examples/s]14177 examples [00:16, 910.87 examples/s]14269 examples [00:16, 908.33 examples/s]14360 examples [00:16, 864.26 examples/s]14452 examples [00:16, 877.93 examples/s]14541 examples [00:16, 865.86 examples/s]14628 examples [00:17, 863.80 examples/s]14715 examples [00:17, 857.25 examples/s]14801 examples [00:17, 855.49 examples/s]14893 examples [00:17, 873.09 examples/s]14982 examples [00:17, 875.65 examples/s]15070 examples [00:17, 870.96 examples/s]15158 examples [00:17, 869.45 examples/s]15246 examples [00:17, 869.61 examples/s]15334 examples [00:17, 863.26 examples/s]15421 examples [00:17, 863.60 examples/s]15516 examples [00:18, 885.09 examples/s]15609 examples [00:18, 895.37 examples/s]15701 examples [00:18, 900.76 examples/s]15792 examples [00:18, 882.70 examples/s]15881 examples [00:18, 857.84 examples/s]15968 examples [00:18, 855.95 examples/s]16056 examples [00:18, 861.78 examples/s]16146 examples [00:18, 870.97 examples/s]16236 examples [00:18, 879.43 examples/s]16328 examples [00:18, 890.26 examples/s]16418 examples [00:19, 882.38 examples/s]16508 examples [00:19, 886.94 examples/s]16601 examples [00:19, 897.42 examples/s]16693 examples [00:19, 901.33 examples/s]16784 examples [00:19, 888.64 examples/s]16873 examples [00:19, 882.00 examples/s]16962 examples [00:19, 864.25 examples/s]17059 examples [00:19, 893.10 examples/s]17156 examples [00:19, 913.85 examples/s]17248 examples [00:20, 887.28 examples/s]17339 examples [00:20, 890.67 examples/s]17429 examples [00:20, 877.30 examples/s]17517 examples [00:20, 852.92 examples/s]17603 examples [00:20, 846.20 examples/s]17688 examples [00:20, 838.01 examples/s]17773 examples [00:20, 840.93 examples/s]17863 examples [00:20, 857.71 examples/s]17952 examples [00:20, 866.36 examples/s]18039 examples [00:20, 854.11 examples/s]18128 examples [00:21, 864.35 examples/s]18220 examples [00:21, 879.22 examples/s]18311 examples [00:21, 886.45 examples/s]18401 examples [00:21, 889.04 examples/s]18491 examples [00:21, 890.96 examples/s]18581 examples [00:21, 880.38 examples/s]18670 examples [00:21, 879.21 examples/s]18760 examples [00:21, 883.97 examples/s]18849 examples [00:21, 874.12 examples/s]18937 examples [00:21, 864.02 examples/s]19024 examples [00:22, 863.52 examples/s]19111 examples [00:22, 856.86 examples/s]19197 examples [00:22, 856.89 examples/s]19284 examples [00:22, 860.00 examples/s]19371 examples [00:22, 857.21 examples/s]19457 examples [00:22, 857.69 examples/s]19547 examples [00:22, 865.69 examples/s]19641 examples [00:22, 885.15 examples/s]19732 examples [00:22, 891.91 examples/s]19822 examples [00:22, 882.54 examples/s]19911 examples [00:23, 870.28 examples/s]19999 examples [00:23, 853.98 examples/s]20085 examples [00:23, 779.81 examples/s]20165 examples [00:23, 784.97 examples/s]20245 examples [00:23, 755.27 examples/s]20326 examples [00:23, 769.51 examples/s]20413 examples [00:23, 796.39 examples/s]20498 examples [00:23, 810.48 examples/s]20584 examples [00:23, 823.85 examples/s]20667 examples [00:24, 811.47 examples/s]20752 examples [00:24, 820.14 examples/s]20835 examples [00:24, 814.25 examples/s]20920 examples [00:24, 822.51 examples/s]21007 examples [00:24, 820.60 examples/s]21090 examples [00:24, 805.98 examples/s]21175 examples [00:24, 816.35 examples/s]21261 examples [00:24, 827.45 examples/s]21350 examples [00:24, 844.18 examples/s]21435 examples [00:24, 811.08 examples/s]21518 examples [00:25, 815.03 examples/s]21600 examples [00:25, 781.20 examples/s]21685 examples [00:25, 798.10 examples/s]21773 examples [00:25, 818.66 examples/s]21860 examples [00:25, 833.00 examples/s]21944 examples [00:25, 831.85 examples/s]22032 examples [00:25, 844.18 examples/s]22117 examples [00:25, 840.45 examples/s]22204 examples [00:25, 848.31 examples/s]22296 examples [00:26, 868.06 examples/s]22389 examples [00:26, 884.47 examples/s]22479 examples [00:26, 886.39 examples/s]22569 examples [00:26, 889.64 examples/s]22660 examples [00:26, 895.54 examples/s]22753 examples [00:26, 903.71 examples/s]22844 examples [00:26, 889.71 examples/s]22934 examples [00:26, 883.29 examples/s]23023 examples [00:26, 864.32 examples/s]23114 examples [00:26, 875.12 examples/s]23202 examples [00:27, 865.18 examples/s]23292 examples [00:27, 873.11 examples/s]23386 examples [00:27, 891.89 examples/s]23479 examples [00:27, 900.91 examples/s]23570 examples [00:27, 901.75 examples/s]23661 examples [00:27, 896.92 examples/s]23751 examples [00:27, 876.90 examples/s]23839 examples [00:27, 852.88 examples/s]23925 examples [00:27, 851.55 examples/s]24012 examples [00:27, 855.75 examples/s]24098 examples [00:28, 851.76 examples/s]24184 examples [00:28, 847.60 examples/s]24269 examples [00:28, 846.41 examples/s]24358 examples [00:28, 856.25 examples/s]24444 examples [00:28, 853.88 examples/s]24530 examples [00:28, 816.90 examples/s]24617 examples [00:28, 831.83 examples/s]24704 examples [00:28, 841.42 examples/s]24789 examples [00:28, 843.32 examples/s]24874 examples [00:29, 805.80 examples/s]24964 examples [00:29, 830.32 examples/s]25057 examples [00:29, 856.10 examples/s]25148 examples [00:29, 871.04 examples/s]25238 examples [00:29, 876.76 examples/s]25327 examples [00:29, 876.27 examples/s]25415 examples [00:29, 870.95 examples/s]25504 examples [00:29, 869.51 examples/s]25596 examples [00:29, 882.70 examples/s]25691 examples [00:29, 901.20 examples/s]25782 examples [00:30, 898.51 examples/s]25872 examples [00:30, 861.63 examples/s]25959 examples [00:30, 858.20 examples/s]26046 examples [00:30, 856.45 examples/s]26136 examples [00:30, 866.80 examples/s]26225 examples [00:30, 872.50 examples/s]26313 examples [00:30, 859.81 examples/s]26401 examples [00:30, 863.65 examples/s]26492 examples [00:30, 874.41 examples/s]26580 examples [00:30, 869.46 examples/s]26668 examples [00:31, 870.13 examples/s]26756 examples [00:31, 864.20 examples/s]26843 examples [00:31, 859.20 examples/s]26929 examples [00:31, 858.94 examples/s]27015 examples [00:31, 848.28 examples/s]27101 examples [00:31, 850.87 examples/s]27187 examples [00:31, 816.07 examples/s]27269 examples [00:31, 805.47 examples/s]27350 examples [00:31, 790.82 examples/s]27440 examples [00:31, 819.31 examples/s]27529 examples [00:32, 838.72 examples/s]27614 examples [00:32, 841.63 examples/s]27701 examples [00:32, 848.63 examples/s]27789 examples [00:32, 856.39 examples/s]27879 examples [00:32, 867.27 examples/s]27966 examples [00:32, 848.20 examples/s]28052 examples [00:32, 824.89 examples/s]28136 examples [00:32, 828.48 examples/s]28223 examples [00:32, 839.90 examples/s]28308 examples [00:33, 840.62 examples/s]28393 examples [00:33, 815.05 examples/s]28475 examples [00:33, 802.40 examples/s]28560 examples [00:33, 814.23 examples/s]28647 examples [00:33, 829.95 examples/s]28731 examples [00:33, 814.27 examples/s]28813 examples [00:33, 774.73 examples/s]28898 examples [00:33, 793.85 examples/s]28989 examples [00:33, 824.10 examples/s]29075 examples [00:33, 832.36 examples/s]29165 examples [00:34, 849.33 examples/s]29256 examples [00:34, 864.37 examples/s]29347 examples [00:34, 874.97 examples/s]29439 examples [00:34, 886.94 examples/s]29528 examples [00:34, 884.46 examples/s]29620 examples [00:34, 894.06 examples/s]29710 examples [00:34, 869.96 examples/s]29798 examples [00:34, 864.19 examples/s]29887 examples [00:34, 870.87 examples/s]29976 examples [00:34, 875.92 examples/s]30064 examples [00:35, 808.76 examples/s]30146 examples [00:35, 794.17 examples/s]30227 examples [00:35, 781.30 examples/s]30315 examples [00:35, 806.54 examples/s]30403 examples [00:35, 827.18 examples/s]30487 examples [00:35, 822.08 examples/s]30575 examples [00:35, 836.60 examples/s]30662 examples [00:35, 845.21 examples/s]30747 examples [00:35, 843.57 examples/s]30832 examples [00:36, 843.72 examples/s]30922 examples [00:36, 857.95 examples/s]31011 examples [00:36, 865.37 examples/s]31102 examples [00:36, 875.50 examples/s]31194 examples [00:36, 887.12 examples/s]31286 examples [00:36, 894.48 examples/s]31376 examples [00:36, 893.35 examples/s]31468 examples [00:36, 901.11 examples/s]31559 examples [00:36, 902.07 examples/s]31650 examples [00:36, 881.93 examples/s]31742 examples [00:37, 891.40 examples/s]31834 examples [00:37, 897.11 examples/s]31926 examples [00:37, 901.48 examples/s]32018 examples [00:37, 904.13 examples/s]32109 examples [00:37, 881.98 examples/s]32199 examples [00:37, 885.55 examples/s]32288 examples [00:37, 883.37 examples/s]32377 examples [00:37, 880.50 examples/s]32467 examples [00:37, 883.31 examples/s]32560 examples [00:37, 894.50 examples/s]32652 examples [00:38, 900.10 examples/s]32744 examples [00:38, 904.53 examples/s]32835 examples [00:38, 903.49 examples/s]32926 examples [00:38, 883.34 examples/s]33015 examples [00:38, 882.89 examples/s]33104 examples [00:38, 819.33 examples/s]33187 examples [00:38, 804.13 examples/s]33269 examples [00:38, 797.98 examples/s]33354 examples [00:38, 811.81 examples/s]33437 examples [00:39, 816.05 examples/s]33519 examples [00:39, 813.26 examples/s]33603 examples [00:39, 820.02 examples/s]33686 examples [00:39, 822.80 examples/s]33777 examples [00:39, 846.32 examples/s]33868 examples [00:39, 863.04 examples/s]33960 examples [00:39, 878.22 examples/s]34054 examples [00:39, 895.19 examples/s]34148 examples [00:39, 905.68 examples/s]34244 examples [00:39, 920.36 examples/s]34338 examples [00:40, 926.08 examples/s]34431 examples [00:40, 924.55 examples/s]34524 examples [00:40, 923.43 examples/s]34617 examples [00:40, 865.66 examples/s]34705 examples [00:40, 859.46 examples/s]34792 examples [00:40, 840.45 examples/s]34888 examples [00:40, 871.72 examples/s]34985 examples [00:40, 898.24 examples/s]35076 examples [00:40, 881.50 examples/s]35165 examples [00:40, 883.63 examples/s]35257 examples [00:41, 893.55 examples/s]35347 examples [00:41, 855.50 examples/s]35434 examples [00:41, 854.77 examples/s]35520 examples [00:41, 849.35 examples/s]35611 examples [00:41, 865.33 examples/s]35700 examples [00:41, 871.47 examples/s]35790 examples [00:41, 878.01 examples/s]35879 examples [00:41, 879.51 examples/s]35968 examples [00:41, 858.25 examples/s]36055 examples [00:41, 843.62 examples/s]36151 examples [00:42, 873.64 examples/s]36246 examples [00:42, 892.76 examples/s]36341 examples [00:42, 908.59 examples/s]36433 examples [00:42, 911.89 examples/s]36526 examples [00:42, 914.84 examples/s]36618 examples [00:42, 915.67 examples/s]36714 examples [00:42, 926.89 examples/s]36810 examples [00:42, 934.58 examples/s]36908 examples [00:42, 946.48 examples/s]37004 examples [00:43, 949.03 examples/s]37100 examples [00:43, 952.11 examples/s]37198 examples [00:43, 958.82 examples/s]37295 examples [00:43, 959.00 examples/s]37391 examples [00:43, 941.69 examples/s]37486 examples [00:43, 930.03 examples/s]37580 examples [00:43, 897.90 examples/s]37671 examples [00:43, 853.76 examples/s]37761 examples [00:43, 865.17 examples/s]37853 examples [00:43, 880.55 examples/s]37946 examples [00:44, 893.95 examples/s]38042 examples [00:44, 911.28 examples/s]38134 examples [00:44, 909.48 examples/s]38226 examples [00:44, 906.65 examples/s]38317 examples [00:44, 896.46 examples/s]38407 examples [00:44, 876.66 examples/s]38495 examples [00:44, 864.60 examples/s]38582 examples [00:44, 853.95 examples/s]38668 examples [00:44, 847.41 examples/s]38759 examples [00:44, 864.88 examples/s]38846 examples [00:45, 855.56 examples/s]38932 examples [00:45, 855.91 examples/s]39018 examples [00:45, 853.61 examples/s]39104 examples [00:45, 850.91 examples/s]39194 examples [00:45, 864.56 examples/s]39281 examples [00:45, 853.01 examples/s]39367 examples [00:45, 849.17 examples/s]39452 examples [00:45, 847.06 examples/s]39537 examples [00:45, 834.92 examples/s]39621 examples [00:46, 833.68 examples/s]39705 examples [00:46, 826.09 examples/s]39792 examples [00:46, 836.36 examples/s]39877 examples [00:46, 839.50 examples/s]39961 examples [00:46, 828.94 examples/s]40044 examples [00:46, 775.64 examples/s]40123 examples [00:46, 765.14 examples/s]40214 examples [00:46, 802.17 examples/s]40305 examples [00:46, 829.74 examples/s]40392 examples [00:46, 839.01 examples/s]40480 examples [00:47, 848.21 examples/s]40569 examples [00:47, 859.07 examples/s]40656 examples [00:47, 856.42 examples/s]40742 examples [00:47, 853.29 examples/s]40828 examples [00:47, 823.32 examples/s]40916 examples [00:47, 838.32 examples/s]41002 examples [00:47, 843.65 examples/s]41087 examples [00:47, 826.70 examples/s]41170 examples [00:47, 820.95 examples/s]41255 examples [00:47, 827.34 examples/s]41344 examples [00:48, 843.16 examples/s]41429 examples [00:48, 839.95 examples/s]41515 examples [00:48, 843.32 examples/s]41600 examples [00:48, 830.56 examples/s]41684 examples [00:48, 830.46 examples/s]41768 examples [00:48, 818.84 examples/s]41850 examples [00:48, 806.10 examples/s]41931 examples [00:48, 801.71 examples/s]42014 examples [00:48, 809.27 examples/s]42103 examples [00:49, 831.27 examples/s]42196 examples [00:49, 858.17 examples/s]42287 examples [00:49, 871.57 examples/s]42382 examples [00:49, 892.61 examples/s]42475 examples [00:49, 902.03 examples/s]42569 examples [00:49, 912.97 examples/s]42661 examples [00:49, 914.66 examples/s]42753 examples [00:49, 910.21 examples/s]42845 examples [00:49, 903.81 examples/s]42936 examples [00:49, 904.00 examples/s]43027 examples [00:50, 905.55 examples/s]43121 examples [00:50, 912.90 examples/s]43215 examples [00:50, 919.33 examples/s]43307 examples [00:50, 918.68 examples/s]43399 examples [00:50, 888.09 examples/s]43489 examples [00:50, 841.78 examples/s]43574 examples [00:50, 835.90 examples/s]43660 examples [00:50, 839.65 examples/s]43747 examples [00:50, 846.35 examples/s]43832 examples [00:50, 843.85 examples/s]43922 examples [00:51, 858.60 examples/s]44011 examples [00:51, 865.56 examples/s]44103 examples [00:51, 877.14 examples/s]44191 examples [00:51, 862.10 examples/s]44278 examples [00:51, 854.98 examples/s]44368 examples [00:51, 866.87 examples/s]44461 examples [00:51, 883.24 examples/s]44550 examples [00:51, 884.77 examples/s]44639 examples [00:51, 855.60 examples/s]44725 examples [00:51, 817.85 examples/s]44808 examples [00:52, 808.18 examples/s]44895 examples [00:52, 825.20 examples/s]44982 examples [00:52, 837.44 examples/s]45069 examples [00:52, 845.99 examples/s]45155 examples [00:52, 850.08 examples/s]45241 examples [00:52, 852.00 examples/s]45327 examples [00:52, 808.39 examples/s]45414 examples [00:52, 824.36 examples/s]45504 examples [00:52, 845.49 examples/s]45596 examples [00:53, 864.13 examples/s]45683 examples [00:53, 852.88 examples/s]45769 examples [00:53, 846.80 examples/s]45854 examples [00:53, 837.46 examples/s]45938 examples [00:53, 836.79 examples/s]46025 examples [00:53, 843.98 examples/s]46116 examples [00:53, 860.62 examples/s]46203 examples [00:53, 840.47 examples/s]46288 examples [00:53, 835.20 examples/s]46378 examples [00:53, 852.99 examples/s]46470 examples [00:54, 870.47 examples/s]46558 examples [00:54, 851.46 examples/s]46644 examples [00:54, 838.41 examples/s]46731 examples [00:54, 847.35 examples/s]46816 examples [00:54, 845.46 examples/s]46901 examples [00:54, 833.22 examples/s]46985 examples [00:54, 835.21 examples/s]47072 examples [00:54, 845.17 examples/s]47159 examples [00:54, 850.78 examples/s]47248 examples [00:54, 859.77 examples/s]47335 examples [00:55, 860.44 examples/s]47422 examples [00:55, 857.13 examples/s]47511 examples [00:55, 866.27 examples/s]47598 examples [00:55, 854.87 examples/s]47684 examples [00:55, 835.51 examples/s]47775 examples [00:55, 853.90 examples/s]47864 examples [00:55, 863.07 examples/s]47951 examples [00:55, 845.24 examples/s]48040 examples [00:55, 857.48 examples/s]48126 examples [00:56, 848.09 examples/s]48211 examples [00:56, 841.09 examples/s]48301 examples [00:56, 856.02 examples/s]48387 examples [00:56, 853.97 examples/s]48477 examples [00:56, 865.23 examples/s]48567 examples [00:56, 874.67 examples/s]48656 examples [00:56, 877.53 examples/s]48750 examples [00:56, 893.91 examples/s]48840 examples [00:56, 893.83 examples/s]48930 examples [00:56, 893.71 examples/s]49020 examples [00:57, 870.35 examples/s]49111 examples [00:57, 881.10 examples/s]49200 examples [00:57, 845.93 examples/s]49289 examples [00:57, 858.21 examples/s]49377 examples [00:57, 864.31 examples/s]49464 examples [00:57, 839.81 examples/s]49555 examples [00:57, 858.28 examples/s]49642 examples [00:57, 859.00 examples/s]49729 examples [00:57, 850.04 examples/s]49817 examples [00:57, 857.71 examples/s]49903 examples [00:58, 857.14 examples/s]49989 examples [00:58, 856.80 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6821/50000 [00:00<00:00, 68208.44 examples/s] 37%|███▋      | 18361/50000 [00:00<00:00, 77746.34 examples/s] 60%|██████    | 30034/50000 [00:00<00:00, 86402.01 examples/s] 83%|████████▎ | 41432/50000 [00:00<00:00, 93164.01 examples/s]                                                               0 examples [00:00, ? examples/s]65 examples [00:00, 641.90 examples/s]139 examples [00:00, 667.31 examples/s]223 examples [00:00, 710.06 examples/s]309 examples [00:00, 748.50 examples/s]399 examples [00:00, 786.98 examples/s]489 examples [00:00, 816.17 examples/s]581 examples [00:00, 844.76 examples/s]666 examples [00:00, 845.30 examples/s]748 examples [00:00, 649.40 examples/s]837 examples [00:01, 706.50 examples/s]929 examples [00:01, 758.02 examples/s]1022 examples [00:01, 801.63 examples/s]1116 examples [00:01, 837.45 examples/s]1212 examples [00:01, 869.83 examples/s]1305 examples [00:01, 886.01 examples/s]1396 examples [00:01, 885.76 examples/s]1486 examples [00:01, 870.09 examples/s]1574 examples [00:01, 849.59 examples/s]1668 examples [00:02, 873.37 examples/s]1763 examples [00:02, 894.58 examples/s]1860 examples [00:02, 914.69 examples/s]1957 examples [00:02, 929.53 examples/s]2051 examples [00:02, 920.69 examples/s]2144 examples [00:02, 910.11 examples/s]2236 examples [00:02, 892.71 examples/s]2328 examples [00:02, 899.14 examples/s]2419 examples [00:02, 889.64 examples/s]2509 examples [00:02, 884.42 examples/s]2604 examples [00:03, 903.04 examples/s]2699 examples [00:03, 916.11 examples/s]2791 examples [00:03, 911.61 examples/s]2885 examples [00:03, 919.46 examples/s]2978 examples [00:03, 910.39 examples/s]3070 examples [00:03, 874.96 examples/s]3161 examples [00:03, 883.21 examples/s]3259 examples [00:03, 908.77 examples/s]3352 examples [00:03, 912.22 examples/s]3444 examples [00:03, 904.29 examples/s]3538 examples [00:04, 912.31 examples/s]3633 examples [00:04, 921.08 examples/s]3726 examples [00:04, 923.11 examples/s]3821 examples [00:04, 931.00 examples/s]3915 examples [00:04, 924.45 examples/s]4010 examples [00:04, 931.11 examples/s]4107 examples [00:04, 940.40 examples/s]4202 examples [00:04, 941.00 examples/s]4299 examples [00:04, 948.22 examples/s]4394 examples [00:04, 925.07 examples/s]4487 examples [00:05, 913.26 examples/s]4579 examples [00:05, 870.78 examples/s]4667 examples [00:05, 845.25 examples/s]4756 examples [00:05, 857.22 examples/s]4843 examples [00:05, 857.56 examples/s]4934 examples [00:05, 870.85 examples/s]5022 examples [00:05, 868.73 examples/s]5110 examples [00:05, 868.27 examples/s]5197 examples [00:05, 850.89 examples/s]5283 examples [00:06, 850.24 examples/s]5372 examples [00:06, 859.49 examples/s]5460 examples [00:06, 862.17 examples/s]5547 examples [00:06, 855.72 examples/s]5638 examples [00:06, 870.90 examples/s]5726 examples [00:06, 870.19 examples/s]5819 examples [00:06, 885.50 examples/s]5912 examples [00:06, 897.09 examples/s]6002 examples [00:06, 883.16 examples/s]6095 examples [00:06, 896.16 examples/s]6186 examples [00:07, 900.03 examples/s]6281 examples [00:07, 912.43 examples/s]6376 examples [00:07, 922.41 examples/s]6469 examples [00:07, 891.77 examples/s]6559 examples [00:07, 887.67 examples/s]6648 examples [00:07, 880.92 examples/s]6737 examples [00:07, 882.68 examples/s]6826 examples [00:07, 878.94 examples/s]6915 examples [00:07, 880.45 examples/s]7004 examples [00:07, 877.33 examples/s]7092 examples [00:08, 861.94 examples/s]7180 examples [00:08, 865.23 examples/s]7269 examples [00:08, 869.89 examples/s]7357 examples [00:08, 871.57 examples/s]7445 examples [00:08, 859.96 examples/s]7532 examples [00:08, 835.66 examples/s]7623 examples [00:08, 855.64 examples/s]7712 examples [00:08, 865.55 examples/s]7799 examples [00:08, 865.72 examples/s]7891 examples [00:09, 880.76 examples/s]7981 examples [00:09, 885.25 examples/s]8072 examples [00:09, 891.55 examples/s]8162 examples [00:09, 870.48 examples/s]8250 examples [00:09, 864.60 examples/s]8340 examples [00:09, 874.52 examples/s]8431 examples [00:09, 884.02 examples/s]8522 examples [00:09, 889.59 examples/s]8614 examples [00:09, 897.19 examples/s]8704 examples [00:09, 889.67 examples/s]8797 examples [00:10, 900.14 examples/s]8888 examples [00:10, 899.96 examples/s]8979 examples [00:10, 889.98 examples/s]9069 examples [00:10, 892.23 examples/s]9163 examples [00:10, 903.78 examples/s]9254 examples [00:10, 890.70 examples/s]9344 examples [00:10, 889.77 examples/s]9434 examples [00:10, 891.89 examples/s]9524 examples [00:10, 891.53 examples/s]9614 examples [00:10, 886.67 examples/s]9703 examples [00:11, 884.75 examples/s]9794 examples [00:11, 891.80 examples/s]9884 examples [00:11, 881.54 examples/s]9973 examples [00:11, 878.07 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteOR151G/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteOR151G/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f249ed2b950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f249ed2b950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f249ed2b950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2504fb9b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2428184f98>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f249ed2b950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f249ed2b950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f249ed2b950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f249ed18c50>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f24850ea0f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f24166cf268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f24166cf268> 

  function with postional parmater data_info <function split_train_valid at 0x7f24166cf268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f24166cf378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f24166cf378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f24166cf378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=7dc5c3bfa0347057566be3d7420f2aba8a9994b80e5624ced9819b74bb65e036
  Stored in directory: /tmp/pip-ephem-wheel-cache-w7jo3ib2/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:01:58, 10.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:21:17, 14.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:30:06, 20.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:03:32, 29.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:37:36, 42.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.83M/862M [00:01<3:54:41, 60.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:43:48, 86.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:01<1:54:00, 123kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.8M/862M [00:01<1:19:37, 176kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.6M/862M [00:01<55:27, 251kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 33.2M/862M [00:01<38:37, 358kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.1M/862M [00:02<27:05, 508kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.0M/862M [00:02<18:53, 723kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.0M/862M [00:02<13:20, 1.02MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.8M/862M [00:02<09:20, 1.45MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:03<08:05, 1.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<07:33, 1.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<07:27, 1.80MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<05:46, 2.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<06:21, 2.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<06:12, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<04:42, 2.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:55, 2.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<07:14, 1.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<05:49, 2.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:09<04:14, 3.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<14:04, 940kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<11:14, 1.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<08:12, 1.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:12<08:40, 1.52MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<08:50, 1.49MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:13<06:53, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:13<04:57, 2.64MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<33:42, 388kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<24:56, 524kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:15<17:45, 735kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<15:25, 844kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:17<13:33, 960kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:17<10:04, 1.29MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:17<07:11, 1.80MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<13:36, 951kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<10:51, 1.19MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:19<07:55, 1.63MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<08:33, 1.50MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<08:35, 1.50MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:21<06:39, 1.93MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<06:43, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<06:00, 2.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:23<04:31, 2.82MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<06:09, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<06:53, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:24<05:28, 2.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<03:58, 3.18MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:33:21, 136kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:06:25, 191kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<46:42, 271kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<32:48, 384kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<50:44, 248kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<38:04, 331kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<27:15, 462kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<21:02, 596kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<15:59, 783kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<11:26, 1.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<08:44, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<9:15:33, 22.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<6:29:13, 32.0kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<4:31:36, 45.7kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<3:19:49, 62.1kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<2:22:27, 87.0kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<1:40:13, 124kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<1:10:02, 176kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:17:27, 159kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<55:28, 222kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<39:03, 315kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<30:07, 407kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<22:19, 549kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<15:54, 769kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<13:57, 874kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<12:24, 982kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<09:17, 1.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<06:36, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<20:45, 584kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<15:47, 767kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<11:20, 1.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:44, 1.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:00, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<07:37, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<05:25, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<44:55, 267kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<32:42, 366kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<23:09, 516kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<18:50, 632kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<15:42, 758kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<11:32, 1.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<08:17, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<09:11, 1.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:41, 1.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:40, 2.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:37, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:53, 2.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:21, 2.69MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:42, 2.05MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:28, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:08, 2.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:43, 3.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<31:49, 366kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<23:29, 495kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<16:43, 694kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<14:16, 811kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<12:26, 930kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<09:13, 1.25MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<06:34, 1.75MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<11:36, 991kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<09:22, 1.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<06:48, 1.69MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<07:17, 1.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:31, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:46, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<04:10, 2.72MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<09:26, 1.20MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:51, 1.45MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:47, 1.96MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:34, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:59, 1.61MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:29, 2.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<03:57, 2.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<28:46, 390kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<21:19, 526kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<15:09, 739kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<13:05, 852kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<11:34, 963kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<08:37, 1.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<06:09, 1.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<10:14, 1.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<08:20, 1.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<06:04, 1.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:44, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:03, 1.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:24, 2.03MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<03:56, 2.78MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:10, 1.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:10, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:34, 2.39MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:39, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:15, 1.74MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<04:57, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:35, 3.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<35:02, 309kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<25:39, 421kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<18:12, 592kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<15:07, 710kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<12:51, 835kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<09:28, 1.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<06:45, 1.58MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<10:27, 1.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:26, 1.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<06:08, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:41, 1.58MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:55, 1.53MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<05:21, 1.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<03:50, 2.74MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<20:56, 503kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<15:46, 668kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<11:17, 931kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<10:14, 1.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<09:23, 1.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<07:01, 1.49MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<05:03, 2.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<08:19, 1.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:57, 1.49MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<05:05, 2.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:10, 2.47MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<7:28:56, 23.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<5:14:27, 32.8kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<3:39:12, 46.9kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<2:45:21, 62.1kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<1:57:53, 87.1kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<1:22:53, 124kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<57:57, 176kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<45:17, 225kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<32:45, 311kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<23:06, 440kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<18:24, 550kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<15:01, 674kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<10:57, 922kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<07:46, 1.30MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<11:40, 861kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<09:14, 1.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<06:43, 1.49MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<06:56, 1.44MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<06:58, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:24, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<03:53, 2.55MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<34:21, 289kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<24:55, 398kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<17:39, 560kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<14:33, 677kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<12:16, 802kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<09:06, 1.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<06:27, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<27:09, 360kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<20:02, 488kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<14:12, 687kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<12:09, 800kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<10:34, 918kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<07:49, 1.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<05:35, 1.73MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<08:32, 1.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<07:00, 1.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:07, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<05:46, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:52, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<03:39, 2.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:45, 2.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:20, 1.78MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:09, 2.28MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:01, 3.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<07:41, 1.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:23, 1.48MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [01:59<04:40, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<05:22, 1.74MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:44, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:26, 2.11MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:12, 2.90MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<08:25, 1.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<06:53, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:00, 1.85MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:36, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<05:53, 1.57MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:31, 2.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:17, 2.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<06:46, 1.35MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<05:42, 1.60MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:11, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:59, 1.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<05:19, 1.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:08, 2.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<02:58, 3.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<14:11, 636kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<10:53, 829kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:50, 1.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<07:29, 1.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<07:04, 1.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:20, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:50, 2.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<07:55, 1.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:29, 1.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:44, 1.87MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:17, 1.67MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:38, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<03:27, 2.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:22, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:55, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:49, 2.28MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<02:46, 3.13MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<06:44, 1.29MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<05:29, 1.58MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<04:06, 2.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:47, 1.80MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<05:10, 1.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:02, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<02:55, 2.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<14:41, 582kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<11:10, 764kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<07:59, 1.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<07:29, 1.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<06:07, 1.38MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:26<04:27, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<05:05, 1.65MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:19, 1.94MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<03:14, 2.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<02:46, 3.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<5:57:16, 23.4kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<4:10:11, 33.3kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<2:54:15, 47.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<2:09:08, 64.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<1:32:07, 89.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<1:04:45, 128kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<45:15, 182kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<35:09, 233kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<25:27, 322kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<17:57, 455kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<14:21, 567kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<11:46, 691kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<08:36, 943kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<06:05, 1.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<46:36, 173kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<33:19, 242kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<23:25, 343kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<16:28, 486kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<23:38, 338kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<18:15, 438kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<13:07, 609kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<09:15, 859kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<10:02, 790kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<07:51, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<05:41, 1.39MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<05:44, 1.37MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:51, 1.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<03:34, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:15, 1.83MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:33, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:32, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:46<02:32, 3.04MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<14:26, 535kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<10:55, 706kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<07:47, 987kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<07:09, 1.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:37, 1.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:58, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<03:33, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<06:58, 1.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:40, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:09, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:36, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:49, 1.56MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:46, 1.99MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:43, 2.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<25:36, 291kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<18:34, 401kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<13:07, 565kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<09:16, 796kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<20:38, 358kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<16:02, 460kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<11:35, 635kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<08:09, 896kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<23:04, 317kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<16:56, 431kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<12:00, 606kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<09:58, 725kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<08:32, 847kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<06:20, 1.14MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:37, 1.56MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:49, 1.49MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:08, 1.73MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:03, 2.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:44, 1.90MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:21, 2.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:30, 2.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:20, 2.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:04, 2.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<02:21, 2.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:08<01:43, 4.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<11:29, 606kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<09:31, 730kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<06:58, 997kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<04:56, 1.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<07:55, 870kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<06:16, 1.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:32, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<04:42, 1.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<04:43, 1.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:36, 1.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:36, 2.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:07, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<04:17, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:09, 2.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:44, 1.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:02, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:07, 2.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:14, 2.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<07:04, 935kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<05:39, 1.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:06, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:20, 1.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:25, 1.48MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<03:22, 1.93MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:27, 2.64MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:05, 1.59MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:33, 1.82MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:37, 2.46MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:17, 1.95MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:39, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:53, 2.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:05, 3.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<16:06, 394kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<11:56, 531kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<08:28, 745kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<07:19, 857kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<06:26, 972kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<04:47, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:25, 1.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<05:28, 1.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:28, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<03:15, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:37, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<4:58:21, 20.6kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<3:28:47, 29.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<2:25:05, 41.9kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<1:47:24, 56.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<1:16:25, 79.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<53:39, 113kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<37:26, 161kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<28:41, 209kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<20:40, 290kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<14:33, 410kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<11:29, 516kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<09:18, 638kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<06:48, 869kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<04:47, 1.22MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<17:31, 335kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<12:52, 455kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<09:07, 639kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<07:39, 757kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<06:34, 881kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:51, 1.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<03:26, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<06:18, 909kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<05:00, 1.14MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:38, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:50, 1.48MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:52, 1.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:57, 1.91MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<02:08, 2.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:25, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:41, 1.52MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:41, 2.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:07, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:21, 1.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:35, 2.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<01:52, 2.92MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:51, 1.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:58, 1.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:54, 1.86MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:14, 1.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:24, 1.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:37, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:53, 2.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:59, 1.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:21, 1.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:27, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:54, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:34, 2.04MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<01:55, 2.70MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:30, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:48, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:11, 2.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<01:35, 3.21MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:14, 1.58MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:48, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:05, 2.42MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:35, 1.95MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:50, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:30, 2.00MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:51, 2.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:21, 3.66MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<2:22:50, 34.8kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<1:40:21, 49.5kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<1:10:22, 70.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<48:52, 100kB/s]   .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<38:17, 128kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<27:49, 176kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<19:38, 249kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<13:48, 352kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<10:49, 447kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<08:04, 598kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<05:45, 835kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<05:04, 940kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<04:33, 1.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:23, 1.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:28, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:57, 1.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:33, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<01:53, 2.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:22, 1.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:08, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:36, 2.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:08, 2.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:28, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<01:55, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:24, 3.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:55, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<02:31, 1.78MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:52, 2.38MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:19, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:04, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<01:33, 2.82MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:05, 2.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:22, 1.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<01:53, 2.29MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:21, 3.15MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<11:16, 380kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<08:20, 513kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<05:54, 719kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<05:03, 834kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:25, 951kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<03:18, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<02:20, 1.78MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<25:32, 162kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<18:17, 226kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<12:49, 320kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<09:50, 414kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<07:45, 525kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<05:37, 722kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<03:56, 1.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<12:00, 334kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<08:48, 454kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<06:14, 637kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<05:12, 755kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<04:28, 878kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<03:18, 1.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<02:20, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:56, 981kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:09, 1.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:17, 1.67MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:49, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<2:56:27, 21.5kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<2:03:17, 30.7kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<1:25:27, 43.9kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<1:01:27, 60.7kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<43:46, 85.2kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<30:42, 121kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<21:18, 172kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<17:06, 214kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<12:19, 297kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<08:39, 419kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<06:49, 526kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<05:09, 696kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<03:40, 972kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<03:23, 1.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<03:07, 1.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:20, 1.50MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:39, 2.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<03:21, 1.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:42, 1.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:57, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:08, 1.59MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:10, 1.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:41, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<01:12, 2.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<24:35, 135kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<17:32, 189kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<12:15, 268kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<09:13, 352kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<06:46, 479kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<04:46, 674kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<04:03, 784kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<03:30, 906kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:36, 1.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:50, 1.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<09:05, 342kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<06:40, 465kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<04:42, 655kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<03:56, 772kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<03:23, 894kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:31, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<01:46, 1.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<15:36, 190kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<11:13, 264kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<07:51, 374kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<06:06, 476kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<04:52, 594kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<03:32, 816kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<02:29, 1.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<03:15, 872kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:34, 1.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:50, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:54, 1.45MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:55, 1.44MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:27, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:03, 2.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:47, 1.50MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:32, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:08, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:22, 1.90MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:31, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:10, 2.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<00:51, 3.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:42, 1.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:27, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:04, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:18, 1.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:27, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:08, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<00:48, 2.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<12:05, 200kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<08:42, 277kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<06:05, 393kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<04:42, 499kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<03:46, 621kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:44, 854kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:54, 1.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<02:53, 792kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<02:15, 1.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:36, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:36, 1.37MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:21, 1.63MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:59, 2.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<00:42, 3.02MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<08:28, 253kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<06:08, 348kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<04:18, 492kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<02:59, 696kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<05:09, 402kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<03:49, 542kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:41, 760kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<01:52, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<04:25, 453kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<03:15, 614kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<02:17, 861kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<01:36, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<11:43, 165kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<08:36, 225kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<06:05, 316kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<04:10, 449kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<07:37, 245kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<05:30, 338kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<03:51, 477kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<03:03, 589kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<02:31, 714kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:50, 968kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:16, 1.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<06:07, 283kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<04:27, 387kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<03:07, 546kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<02:30, 663kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:54, 868kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:21, 1.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:16, 1.24MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:14, 1.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:55, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:39, 2.34MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:04, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:54, 1.66MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:39, 2.25MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:47, 1.85MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:50, 1.72MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:38, 2.22MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:27, 3.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:54, 1.52MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:46, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:34, 2.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:41, 1.92MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:36, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:26, 2.86MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:36, 2.08MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:41, 1.82MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:32, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:22, 3.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<03:06, 380kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<02:17, 512kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<01:36, 718kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:20, 833kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:01, 1.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:43, 1.47MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:43, 1.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:43, 1.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:32, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:23, 2.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:43, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:36, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:26, 2.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:30, 1.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:32, 1.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:24, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:17, 2.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:26, 1.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:23, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:17, 2.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:21, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:24, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:19, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:13, 3.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:40, 1.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:32, 1.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:22, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:23, 1.60MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:20, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:14, 2.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:17, 1.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:15, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:11, 2.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:13, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:15, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:08, 3.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<01:06, 380kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:48, 518kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:32, 726kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:25, 836kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:22, 954kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:15, 1.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:09, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:58, 294kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:41, 402kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:26, 566kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:18, 683kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:14, 884kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:08, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:06, 1.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:06, 1.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.71MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:03, 1.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.65MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.21MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.84MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 2.08MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 183/400000 [00:00<03:38, 1829.13it/s]  0%|          | 949/400000 [00:00<02:48, 2370.23it/s]  0%|          | 1665/400000 [00:00<02:14, 2964.89it/s]  1%|          | 2412/400000 [00:00<01:49, 3619.25it/s]  1%|          | 3090/400000 [00:00<01:34, 4207.67it/s]  1%|          | 3732/400000 [00:00<01:24, 4692.20it/s]  1%|          | 4486/400000 [00:00<01:14, 5291.12it/s]  1%|▏         | 5263/400000 [00:00<01:07, 5849.93it/s]  2%|▏         | 6026/400000 [00:00<01:02, 6289.11it/s]  2%|▏         | 6822/400000 [00:01<00:58, 6710.27it/s]  2%|▏         | 7553/400000 [00:01<00:57, 6865.74it/s]  2%|▏         | 8282/400000 [00:01<00:58, 6749.42it/s]  2%|▏         | 8988/400000 [00:01<00:58, 6733.69it/s]  2%|▏         | 9683/400000 [00:01<00:57, 6739.27it/s]  3%|▎         | 10372/400000 [00:01<00:57, 6749.76it/s]  3%|▎         | 11088/400000 [00:01<00:56, 6867.64it/s]  3%|▎         | 11840/400000 [00:01<00:55, 7050.18it/s]  3%|▎         | 12564/400000 [00:01<00:54, 7104.41it/s]  3%|▎         | 13340/400000 [00:01<00:53, 7287.99it/s]  4%|▎         | 14120/400000 [00:02<00:51, 7432.41it/s]  4%|▎         | 14868/400000 [00:02<00:51, 7410.49it/s]  4%|▍         | 15619/400000 [00:02<00:51, 7437.90it/s]  4%|▍         | 16369/400000 [00:02<00:51, 7454.17it/s]  4%|▍         | 17121/400000 [00:02<00:51, 7472.98it/s]  4%|▍         | 17916/400000 [00:02<00:50, 7609.43it/s]  5%|▍         | 18679/400000 [00:02<00:50, 7569.56it/s]  5%|▍         | 19439/400000 [00:02<00:50, 7578.14it/s]  5%|▌         | 20227/400000 [00:02<00:49, 7663.90it/s]  5%|▌         | 20995/400000 [00:02<00:49, 7664.78it/s]  5%|▌         | 21762/400000 [00:03<00:49, 7646.48it/s]  6%|▌         | 22527/400000 [00:03<00:49, 7593.42it/s]  6%|▌         | 23287/400000 [00:03<00:49, 7552.10it/s]  6%|▌         | 24049/400000 [00:03<00:49, 7569.94it/s]  6%|▌         | 24807/400000 [00:03<00:49, 7529.30it/s]  6%|▋         | 25568/400000 [00:03<00:49, 7551.79it/s]  7%|▋         | 26331/400000 [00:03<00:49, 7572.60it/s]  7%|▋         | 27107/400000 [00:03<00:48, 7625.90it/s]  7%|▋         | 27870/400000 [00:03<00:48, 7607.87it/s]  7%|▋         | 28631/400000 [00:03<00:49, 7510.41it/s]  7%|▋         | 29383/400000 [00:04<00:50, 7406.07it/s]  8%|▊         | 30125/400000 [00:04<00:50, 7307.13it/s]  8%|▊         | 30889/400000 [00:04<00:49, 7403.89it/s]  8%|▊         | 31636/400000 [00:04<00:49, 7419.99it/s]  8%|▊         | 32379/400000 [00:04<00:50, 7302.54it/s]  8%|▊         | 33163/400000 [00:04<00:49, 7453.75it/s]  8%|▊         | 33910/400000 [00:04<00:49, 7397.80it/s]  9%|▊         | 34651/400000 [00:04<00:50, 7302.92it/s]  9%|▉         | 35391/400000 [00:04<00:49, 7330.07it/s]  9%|▉         | 36140/400000 [00:04<00:49, 7377.29it/s]  9%|▉         | 36925/400000 [00:05<00:48, 7511.02it/s]  9%|▉         | 37678/400000 [00:05<00:48, 7516.68it/s] 10%|▉         | 38431/400000 [00:05<00:48, 7456.25it/s] 10%|▉         | 39183/400000 [00:05<00:48, 7473.47it/s] 10%|▉         | 39971/400000 [00:05<00:47, 7590.42it/s] 10%|█         | 40731/400000 [00:05<00:50, 7183.33it/s] 10%|█         | 41455/400000 [00:05<00:51, 6944.09it/s] 11%|█         | 42203/400000 [00:05<00:50, 7094.40it/s] 11%|█         | 42936/400000 [00:05<00:49, 7163.21it/s] 11%|█         | 43679/400000 [00:05<00:49, 7239.83it/s] 11%|█         | 44418/400000 [00:06<00:48, 7281.19it/s] 11%|█▏        | 45148/400000 [00:06<00:49, 7204.91it/s] 11%|█▏        | 45914/400000 [00:06<00:48, 7334.58it/s] 12%|█▏        | 46650/400000 [00:06<00:48, 7230.68it/s] 12%|█▏        | 47375/400000 [00:06<00:49, 7193.08it/s] 12%|█▏        | 48096/400000 [00:06<00:48, 7194.00it/s] 12%|█▏        | 48817/400000 [00:06<00:48, 7169.71it/s] 12%|█▏        | 49535/400000 [00:06<00:49, 7106.28it/s] 13%|█▎        | 50247/400000 [00:06<00:49, 6998.71it/s] 13%|█▎        | 50948/400000 [00:07<00:50, 6958.05it/s] 13%|█▎        | 51645/400000 [00:07<00:50, 6844.23it/s] 13%|█▎        | 52331/400000 [00:07<00:51, 6696.00it/s] 13%|█▎        | 53002/400000 [00:07<00:52, 6585.82it/s] 13%|█▎        | 53723/400000 [00:07<00:51, 6761.36it/s] 14%|█▎        | 54419/400000 [00:07<00:50, 6817.07it/s] 14%|█▍        | 55103/400000 [00:07<00:50, 6787.96it/s] 14%|█▍        | 55783/400000 [00:07<00:51, 6657.78it/s] 14%|█▍        | 56457/400000 [00:07<00:51, 6681.38it/s] 14%|█▍        | 57170/400000 [00:07<00:50, 6808.85it/s] 14%|█▍        | 57877/400000 [00:08<00:49, 6884.64it/s] 15%|█▍        | 58567/400000 [00:08<00:50, 6811.27it/s] 15%|█▍        | 59282/400000 [00:08<00:49, 6907.19it/s] 15%|█▌        | 60010/400000 [00:08<00:48, 7013.52it/s] 15%|█▌        | 60713/400000 [00:08<00:49, 6805.39it/s] 15%|█▌        | 61457/400000 [00:08<00:48, 6982.29it/s] 16%|█▌        | 62158/400000 [00:08<00:48, 6928.15it/s] 16%|█▌        | 62853/400000 [00:08<00:49, 6862.53it/s] 16%|█▌        | 63541/400000 [00:08<00:49, 6862.09it/s] 16%|█▌        | 64229/400000 [00:08<00:50, 6677.68it/s] 16%|█▌        | 64941/400000 [00:09<00:49, 6803.82it/s] 16%|█▋        | 65624/400000 [00:09<00:49, 6747.65it/s] 17%|█▋        | 66318/400000 [00:09<00:49, 6803.98it/s] 17%|█▋        | 67026/400000 [00:09<00:48, 6881.91it/s] 17%|█▋        | 67726/400000 [00:09<00:48, 6915.55it/s] 17%|█▋        | 68428/400000 [00:09<00:47, 6944.51it/s] 17%|█▋        | 69123/400000 [00:09<00:48, 6889.00it/s] 17%|█▋        | 69820/400000 [00:09<00:47, 6912.19it/s] 18%|█▊        | 70512/400000 [00:09<00:47, 6874.49it/s] 18%|█▊        | 71212/400000 [00:09<00:47, 6909.70it/s] 18%|█▊        | 71904/400000 [00:10<00:47, 6852.35it/s] 18%|█▊        | 72590/400000 [00:10<00:47, 6846.14it/s] 18%|█▊        | 73332/400000 [00:10<00:46, 7007.14it/s] 19%|█▊        | 74095/400000 [00:10<00:45, 7182.01it/s] 19%|█▊        | 74856/400000 [00:10<00:44, 7304.48it/s] 19%|█▉        | 75589/400000 [00:10<00:45, 7132.08it/s] 19%|█▉        | 76305/400000 [00:10<00:47, 6791.52it/s] 19%|█▉        | 76990/400000 [00:10<00:48, 6721.29it/s] 19%|█▉        | 77672/400000 [00:10<00:47, 6749.65it/s] 20%|█▉        | 78371/400000 [00:11<00:47, 6819.04it/s] 20%|█▉        | 79079/400000 [00:11<00:46, 6892.60it/s] 20%|█▉        | 79770/400000 [00:11<00:46, 6845.76it/s] 20%|██        | 80526/400000 [00:11<00:45, 7045.15it/s] 20%|██        | 81240/400000 [00:11<00:45, 7071.71it/s] 21%|██        | 82012/400000 [00:11<00:43, 7254.41it/s] 21%|██        | 82786/400000 [00:11<00:42, 7393.24it/s] 21%|██        | 83529/400000 [00:11<00:42, 7402.61it/s] 21%|██        | 84293/400000 [00:11<00:42, 7470.45it/s] 21%|██▏       | 85057/400000 [00:11<00:41, 7519.66it/s] 21%|██▏       | 85810/400000 [00:12<00:43, 7264.63it/s] 22%|██▏       | 86556/400000 [00:12<00:42, 7318.52it/s] 22%|██▏       | 87328/400000 [00:12<00:42, 7432.68it/s] 22%|██▏       | 88074/400000 [00:12<00:42, 7396.11it/s] 22%|██▏       | 88815/400000 [00:12<00:42, 7248.98it/s] 22%|██▏       | 89562/400000 [00:12<00:42, 7313.18it/s] 23%|██▎       | 90321/400000 [00:12<00:41, 7393.92it/s] 23%|██▎       | 91062/400000 [00:12<00:42, 7287.14it/s] 23%|██▎       | 91792/400000 [00:12<00:43, 7101.17it/s] 23%|██▎       | 92504/400000 [00:12<00:43, 7012.85it/s] 23%|██▎       | 93207/400000 [00:13<00:43, 7014.12it/s] 23%|██▎       | 93910/400000 [00:13<00:43, 6993.00it/s] 24%|██▎       | 94611/400000 [00:13<00:43, 6949.89it/s] 24%|██▍       | 95307/400000 [00:13<00:44, 6784.53it/s] 24%|██▍       | 96065/400000 [00:13<00:43, 7004.23it/s] 24%|██▍       | 96849/400000 [00:13<00:41, 7234.77it/s] 24%|██▍       | 97628/400000 [00:13<00:40, 7391.67it/s] 25%|██▍       | 98395/400000 [00:13<00:40, 7471.52it/s] 25%|██▍       | 99145/400000 [00:13<00:40, 7419.90it/s] 25%|██▍       | 99898/400000 [00:13<00:40, 7450.84it/s] 25%|██▌       | 100653/400000 [00:14<00:40, 7478.32it/s] 25%|██▌       | 101411/400000 [00:14<00:39, 7506.37it/s] 26%|██▌       | 102167/400000 [00:14<00:39, 7521.30it/s] 26%|██▌       | 102920/400000 [00:14<00:41, 7194.24it/s] 26%|██▌       | 103643/400000 [00:14<00:42, 6987.97it/s] 26%|██▌       | 104346/400000 [00:14<00:42, 6938.38it/s] 26%|██▋       | 105089/400000 [00:14<00:41, 7076.51it/s] 26%|██▋       | 105811/400000 [00:14<00:41, 7114.93it/s] 27%|██▋       | 106566/400000 [00:14<00:40, 7238.04it/s] 27%|██▋       | 107313/400000 [00:15<00:40, 7305.73it/s] 27%|██▋       | 108045/400000 [00:15<00:40, 7200.63it/s] 27%|██▋       | 108807/400000 [00:15<00:39, 7318.77it/s] 27%|██▋       | 109557/400000 [00:15<00:39, 7371.74it/s] 28%|██▊       | 110296/400000 [00:15<00:40, 7194.72it/s] 28%|██▊       | 111018/400000 [00:15<00:41, 7023.11it/s] 28%|██▊       | 111723/400000 [00:15<00:41, 6865.19it/s] 28%|██▊       | 112412/400000 [00:15<00:43, 6628.44it/s] 28%|██▊       | 113080/400000 [00:15<00:43, 6643.67it/s] 28%|██▊       | 113810/400000 [00:15<00:41, 6827.78it/s] 29%|██▊       | 114538/400000 [00:16<00:41, 6955.59it/s] 29%|██▉       | 115237/400000 [00:16<00:41, 6869.85it/s] 29%|██▉       | 115927/400000 [00:16<00:41, 6870.36it/s] 29%|██▉       | 116616/400000 [00:16<00:42, 6737.14it/s] 29%|██▉       | 117333/400000 [00:16<00:41, 6859.48it/s] 30%|██▉       | 118021/400000 [00:16<00:41, 6846.64it/s] 30%|██▉       | 118707/400000 [00:16<00:41, 6794.60it/s] 30%|██▉       | 119388/400000 [00:16<00:41, 6768.31it/s] 30%|███       | 120066/400000 [00:16<00:41, 6728.37it/s] 30%|███       | 120837/400000 [00:16<00:39, 6995.39it/s] 30%|███       | 121599/400000 [00:17<00:38, 7170.33it/s] 31%|███       | 122320/400000 [00:17<00:41, 6712.51it/s] 31%|███       | 123001/400000 [00:17<00:41, 6740.55it/s] 31%|███       | 123683/400000 [00:17<00:40, 6762.17it/s] 31%|███       | 124433/400000 [00:17<00:39, 6967.78it/s] 31%|███▏      | 125190/400000 [00:17<00:38, 7137.80it/s] 31%|███▏      | 125977/400000 [00:17<00:37, 7342.16it/s] 32%|███▏      | 126742/400000 [00:17<00:36, 7429.68it/s] 32%|███▏      | 127489/400000 [00:17<00:37, 7228.47it/s] 32%|███▏      | 128216/400000 [00:18<00:38, 7131.88it/s] 32%|███▏      | 128932/400000 [00:18<00:38, 7031.14it/s] 32%|███▏      | 129638/400000 [00:18<00:39, 6909.19it/s] 33%|███▎      | 130331/400000 [00:18<00:39, 6866.17it/s] 33%|███▎      | 131020/400000 [00:18<00:39, 6831.02it/s] 33%|███▎      | 131793/400000 [00:18<00:37, 7076.94it/s] 33%|███▎      | 132573/400000 [00:18<00:36, 7277.48it/s] 33%|███▎      | 133362/400000 [00:18<00:35, 7449.50it/s] 34%|███▎      | 134125/400000 [00:18<00:35, 7502.11it/s] 34%|███▎      | 134878/400000 [00:18<00:35, 7460.83it/s] 34%|███▍      | 135626/400000 [00:19<00:35, 7460.49it/s] 34%|███▍      | 136374/400000 [00:19<00:36, 7186.58it/s] 34%|███▍      | 137096/400000 [00:19<00:36, 7133.76it/s] 34%|███▍      | 137812/400000 [00:19<00:37, 6978.04it/s] 35%|███▍      | 138513/400000 [00:19<00:38, 6843.61it/s] 35%|███▍      | 139219/400000 [00:19<00:37, 6889.41it/s] 35%|███▍      | 139952/400000 [00:19<00:37, 7015.65it/s] 35%|███▌      | 140681/400000 [00:19<00:36, 7095.06it/s] 35%|███▌      | 141419/400000 [00:19<00:36, 7176.76it/s] 36%|███▌      | 142138/400000 [00:19<00:36, 6999.60it/s] 36%|███▌      | 142840/400000 [00:20<00:36, 6973.48it/s] 36%|███▌      | 143539/400000 [00:20<00:36, 6957.73it/s] 36%|███▌      | 144236/400000 [00:20<00:36, 6922.54it/s] 36%|███▌      | 144953/400000 [00:20<00:36, 6992.10it/s] 36%|███▋      | 145653/400000 [00:20<00:36, 6954.70it/s] 37%|███▋      | 146361/400000 [00:20<00:36, 6988.79it/s] 37%|███▋      | 147061/400000 [00:20<00:36, 6961.97it/s] 37%|███▋      | 147758/400000 [00:20<00:36, 6921.25it/s] 37%|███▋      | 148451/400000 [00:20<00:36, 6837.12it/s] 37%|███▋      | 149136/400000 [00:20<00:37, 6737.48it/s] 37%|███▋      | 149816/400000 [00:21<00:37, 6755.56it/s] 38%|███▊      | 150507/400000 [00:21<00:36, 6798.45it/s] 38%|███▊      | 151188/400000 [00:21<00:36, 6745.03it/s] 38%|███▊      | 151863/400000 [00:21<00:37, 6561.44it/s] 38%|███▊      | 152554/400000 [00:21<00:37, 6660.15it/s] 38%|███▊      | 153278/400000 [00:21<00:36, 6823.39it/s] 38%|███▊      | 153963/400000 [00:21<00:36, 6825.95it/s] 39%|███▊      | 154663/400000 [00:21<00:35, 6876.34it/s] 39%|███▉      | 155352/400000 [00:21<00:35, 6831.97it/s] 39%|███▉      | 156067/400000 [00:22<00:35, 6922.46it/s] 39%|███▉      | 156840/400000 [00:22<00:34, 7146.40it/s] 39%|███▉      | 157590/400000 [00:22<00:33, 7248.53it/s] 40%|███▉      | 158333/400000 [00:22<00:33, 7300.57it/s] 40%|███▉      | 159065/400000 [00:22<00:33, 7255.16it/s] 40%|███▉      | 159792/400000 [00:22<00:34, 7053.58it/s] 40%|████      | 160500/400000 [00:22<00:34, 6998.31it/s] 40%|████      | 161202/400000 [00:22<00:34, 6907.87it/s] 40%|████      | 161920/400000 [00:22<00:34, 6987.30it/s] 41%|████      | 162620/400000 [00:22<00:34, 6873.78it/s] 41%|████      | 163330/400000 [00:23<00:34, 6938.35it/s] 41%|████      | 164071/400000 [00:23<00:33, 7073.29it/s] 41%|████      | 164824/400000 [00:23<00:32, 7204.12it/s] 41%|████▏     | 165615/400000 [00:23<00:31, 7400.53it/s] 42%|████▏     | 166358/400000 [00:23<00:31, 7390.43it/s] 42%|████▏     | 167099/400000 [00:23<00:32, 7220.14it/s] 42%|████▏     | 167824/400000 [00:23<00:32, 7058.34it/s] 42%|████▏     | 168533/400000 [00:23<00:33, 6957.22it/s] 42%|████▏     | 169236/400000 [00:23<00:33, 6977.72it/s] 42%|████▏     | 169936/400000 [00:23<00:33, 6947.46it/s] 43%|████▎     | 170632/400000 [00:24<00:33, 6810.17it/s] 43%|████▎     | 171315/400000 [00:24<00:33, 6803.51it/s] 43%|████▎     | 172035/400000 [00:24<00:32, 6917.40it/s] 43%|████▎     | 172791/400000 [00:24<00:32, 7098.30it/s] 43%|████▎     | 173582/400000 [00:24<00:30, 7322.02it/s] 44%|████▎     | 174331/400000 [00:24<00:30, 7371.54it/s] 44%|████▍     | 175071/400000 [00:24<00:30, 7298.00it/s] 44%|████▍     | 175803/400000 [00:24<00:30, 7272.18it/s] 44%|████▍     | 176532/400000 [00:24<00:31, 7045.58it/s] 44%|████▍     | 177273/400000 [00:24<00:31, 7150.06it/s] 44%|████▍     | 177991/400000 [00:25<00:31, 7032.39it/s] 45%|████▍     | 178697/400000 [00:25<00:31, 6981.08it/s] 45%|████▍     | 179397/400000 [00:25<00:31, 6961.41it/s] 45%|████▌     | 180108/400000 [00:25<00:31, 7002.94it/s] 45%|████▌     | 180870/400000 [00:25<00:30, 7177.16it/s] 45%|████▌     | 181622/400000 [00:25<00:30, 7274.17it/s] 46%|████▌     | 182419/400000 [00:25<00:29, 7468.18it/s] 46%|████▌     | 183185/400000 [00:25<00:28, 7521.79it/s] 46%|████▌     | 183943/400000 [00:25<00:28, 7538.11it/s] 46%|████▌     | 184748/400000 [00:25<00:28, 7682.35it/s] 46%|████▋     | 185518/400000 [00:26<00:28, 7627.91it/s] 47%|████▋     | 186318/400000 [00:26<00:27, 7735.80it/s] 47%|████▋     | 187120/400000 [00:26<00:27, 7818.18it/s] 47%|████▋     | 187911/400000 [00:26<00:27, 7845.48it/s] 47%|████▋     | 188707/400000 [00:26<00:26, 7877.87it/s] 47%|████▋     | 189496/400000 [00:26<00:28, 7477.72it/s] 48%|████▊     | 190249/400000 [00:26<00:28, 7401.71it/s] 48%|████▊     | 190993/400000 [00:26<00:28, 7366.42it/s] 48%|████▊     | 191733/400000 [00:26<00:28, 7199.80it/s] 48%|████▊     | 192456/400000 [00:27<00:29, 7075.57it/s] 48%|████▊     | 193210/400000 [00:27<00:28, 7206.45it/s] 48%|████▊     | 193986/400000 [00:27<00:27, 7361.38it/s] 49%|████▊     | 194773/400000 [00:27<00:27, 7506.37it/s] 49%|████▉     | 195554/400000 [00:27<00:26, 7594.70it/s] 49%|████▉     | 196316/400000 [00:27<00:26, 7581.78it/s] 49%|████▉     | 197088/400000 [00:27<00:26, 7620.97it/s] 49%|████▉     | 197852/400000 [00:27<00:26, 7620.95it/s] 50%|████▉     | 198672/400000 [00:27<00:25, 7785.45it/s] 50%|████▉     | 199452/400000 [00:27<00:25, 7764.48it/s] 50%|█████     | 200230/400000 [00:28<00:25, 7745.13it/s] 50%|█████     | 201007/400000 [00:28<00:25, 7752.01it/s] 50%|█████     | 201783/400000 [00:28<00:25, 7751.65it/s] 51%|█████     | 202559/400000 [00:28<00:25, 7642.44it/s] 51%|█████     | 203324/400000 [00:28<00:26, 7429.52it/s] 51%|█████     | 204069/400000 [00:28<00:26, 7332.56it/s] 51%|█████     | 204819/400000 [00:28<00:26, 7381.00it/s] 51%|█████▏    | 205653/400000 [00:28<00:25, 7644.57it/s] 52%|█████▏    | 206483/400000 [00:28<00:24, 7828.69it/s] 52%|█████▏    | 207339/400000 [00:28<00:23, 8033.41it/s] 52%|█████▏    | 208146/400000 [00:29<00:23, 8009.66it/s] 52%|█████▏    | 208950/400000 [00:29<00:24, 7937.16it/s] 52%|█████▏    | 209746/400000 [00:29<00:24, 7871.71it/s] 53%|█████▎    | 210535/400000 [00:29<00:24, 7809.52it/s] 53%|█████▎    | 211340/400000 [00:29<00:23, 7878.47it/s] 53%|█████▎    | 212129/400000 [00:29<00:23, 7839.68it/s] 53%|█████▎    | 212922/400000 [00:29<00:23, 7866.30it/s] 53%|█████▎    | 213721/400000 [00:29<00:23, 7899.89it/s] 54%|█████▎    | 214512/400000 [00:29<00:23, 7870.38it/s] 54%|█████▍    | 215300/400000 [00:29<00:23, 7818.26it/s] 54%|█████▍    | 216083/400000 [00:30<00:24, 7559.30it/s] 54%|█████▍    | 216841/400000 [00:30<00:24, 7423.92it/s] 54%|█████▍    | 217586/400000 [00:30<00:24, 7309.34it/s] 55%|█████▍    | 218319/400000 [00:30<00:24, 7310.69it/s] 55%|█████▍    | 219101/400000 [00:30<00:24, 7455.38it/s] 55%|█████▍    | 219866/400000 [00:30<00:23, 7512.25it/s] 55%|█████▌    | 220619/400000 [00:30<00:24, 7362.79it/s] 55%|█████▌    | 221357/400000 [00:30<00:24, 7310.52it/s] 56%|█████▌    | 222090/400000 [00:30<00:24, 7286.92it/s] 56%|█████▌    | 222846/400000 [00:31<00:24, 7366.68it/s] 56%|█████▌    | 223584/400000 [00:31<00:24, 7276.55it/s] 56%|█████▌    | 224313/400000 [00:31<00:24, 7168.86it/s] 56%|█████▋    | 225031/400000 [00:31<00:24, 7122.42it/s] 56%|█████▋    | 225750/400000 [00:31<00:24, 7139.98it/s] 57%|█████▋    | 226474/400000 [00:31<00:24, 7167.69it/s] 57%|█████▋    | 227192/400000 [00:31<00:24, 7082.20it/s] 57%|█████▋    | 227911/400000 [00:31<00:24, 7112.47it/s] 57%|█████▋    | 228696/400000 [00:31<00:23, 7317.53it/s] 57%|█████▋    | 229453/400000 [00:31<00:23, 7389.93it/s] 58%|█████▊    | 230213/400000 [00:32<00:22, 7451.13it/s] 58%|█████▊    | 230996/400000 [00:32<00:22, 7557.11it/s] 58%|█████▊    | 231753/400000 [00:32<00:22, 7522.60it/s] 58%|█████▊    | 232507/400000 [00:32<00:22, 7307.29it/s] 58%|█████▊    | 233240/400000 [00:32<00:23, 7135.83it/s] 58%|█████▊    | 233956/400000 [00:32<00:23, 7085.19it/s] 59%|█████▊    | 234667/400000 [00:32<00:23, 6934.47it/s] 59%|█████▉    | 235363/400000 [00:32<00:23, 6926.81it/s] 59%|█████▉    | 236057/400000 [00:32<00:23, 6891.75it/s] 59%|█████▉    | 236748/400000 [00:32<00:23, 6834.19it/s] 59%|█████▉    | 237463/400000 [00:33<00:23, 6923.24it/s] 60%|█████▉    | 238157/400000 [00:33<00:24, 6610.81it/s] 60%|█████▉    | 238893/400000 [00:33<00:23, 6817.96it/s] 60%|█████▉    | 239579/400000 [00:33<00:23, 6758.48it/s] 60%|██████    | 240270/400000 [00:33<00:23, 6802.84it/s] 60%|██████    | 240996/400000 [00:33<00:22, 6933.27it/s] 60%|██████    | 241702/400000 [00:33<00:22, 6968.64it/s] 61%|██████    | 242401/400000 [00:33<00:23, 6793.22it/s] 61%|██████    | 243100/400000 [00:33<00:22, 6851.01it/s] 61%|██████    | 243787/400000 [00:33<00:23, 6752.57it/s] 61%|██████    | 244469/400000 [00:34<00:22, 6772.51it/s] 61%|██████▏   | 245148/400000 [00:34<00:23, 6563.50it/s] 61%|██████▏   | 245893/400000 [00:34<00:22, 6806.21it/s] 62%|██████▏   | 246649/400000 [00:34<00:21, 7013.04it/s] 62%|██████▏   | 247444/400000 [00:34<00:20, 7267.19it/s] 62%|██████▏   | 248241/400000 [00:34<00:20, 7464.52it/s] 62%|██████▏   | 248993/400000 [00:34<00:20, 7476.20it/s] 62%|██████▏   | 249745/400000 [00:34<00:20, 7438.54it/s] 63%|██████▎   | 250507/400000 [00:34<00:19, 7491.15it/s] 63%|██████▎   | 251258/400000 [00:35<00:20, 7345.51it/s] 63%|██████▎   | 251995/400000 [00:35<00:20, 7220.17it/s] 63%|██████▎   | 252719/400000 [00:35<00:20, 7141.68it/s] 63%|██████▎   | 253442/400000 [00:35<00:20, 7165.83it/s] 64%|██████▎   | 254160/400000 [00:35<00:20, 7164.17it/s] 64%|██████▎   | 254893/400000 [00:35<00:20, 7211.06it/s] 64%|██████▍   | 255615/400000 [00:35<00:20, 7041.50it/s] 64%|██████▍   | 256326/400000 [00:35<00:20, 7057.41it/s] 64%|██████▍   | 257090/400000 [00:35<00:19, 7222.15it/s] 64%|██████▍   | 257823/400000 [00:35<00:19, 7252.42it/s] 65%|██████▍   | 258570/400000 [00:36<00:19, 7314.16it/s] 65%|██████▍   | 259329/400000 [00:36<00:19, 7394.06it/s] 65%|██████▌   | 260070/400000 [00:36<00:19, 7334.79it/s] 65%|██████▌   | 260860/400000 [00:36<00:18, 7493.44it/s] 65%|██████▌   | 261611/400000 [00:36<00:18, 7480.89it/s] 66%|██████▌   | 262393/400000 [00:36<00:18, 7578.33it/s] 66%|██████▌   | 263157/400000 [00:36<00:18, 7594.63it/s] 66%|██████▌   | 263918/400000 [00:36<00:17, 7572.82it/s] 66%|██████▌   | 264716/400000 [00:36<00:17, 7689.68it/s] 66%|██████▋   | 265522/400000 [00:36<00:17, 7795.86it/s] 67%|██████▋   | 266303/400000 [00:37<00:17, 7719.39it/s] 67%|██████▋   | 267077/400000 [00:37<00:17, 7722.48it/s] 67%|██████▋   | 267850/400000 [00:37<00:17, 7631.36it/s] 67%|██████▋   | 268614/400000 [00:37<00:17, 7623.10it/s] 67%|██████▋   | 269377/400000 [00:37<00:17, 7539.52it/s] 68%|██████▊   | 270132/400000 [00:37<00:17, 7300.12it/s] 68%|██████▊   | 270865/400000 [00:37<00:18, 7116.47it/s] 68%|██████▊   | 271580/400000 [00:37<00:18, 7019.88it/s] 68%|██████▊   | 272292/400000 [00:37<00:18, 7048.05it/s] 68%|██████▊   | 272999/400000 [00:37<00:18, 6993.05it/s] 68%|██████▊   | 273700/400000 [00:38<00:18, 6983.68it/s] 69%|██████▊   | 274426/400000 [00:38<00:17, 7063.81it/s] 69%|██████▉   | 275134/400000 [00:38<00:17, 6938.25it/s] 69%|██████▉   | 275829/400000 [00:38<00:18, 6889.50it/s] 69%|██████▉   | 276527/400000 [00:38<00:17, 6914.34it/s] 69%|██████▉   | 277266/400000 [00:38<00:17, 7048.38it/s] 69%|██████▉   | 277998/400000 [00:38<00:17, 7124.60it/s] 70%|██████▉   | 278721/400000 [00:38<00:16, 7155.51it/s] 70%|██████▉   | 279438/400000 [00:38<00:17, 6939.31it/s] 70%|███████   | 280153/400000 [00:38<00:17, 6998.33it/s] 70%|███████   | 280878/400000 [00:39<00:16, 7071.81it/s] 70%|███████   | 281587/400000 [00:39<00:17, 6763.79it/s] 71%|███████   | 282268/400000 [00:39<00:17, 6582.81it/s] 71%|███████   | 282948/400000 [00:39<00:17, 6645.69it/s] 71%|███████   | 283636/400000 [00:39<00:17, 6714.11it/s] 71%|███████   | 284393/400000 [00:39<00:16, 6948.83it/s] 71%|███████▏  | 285092/400000 [00:39<00:16, 6888.21it/s] 71%|███████▏  | 285788/400000 [00:39<00:16, 6907.48it/s] 72%|███████▏  | 286483/400000 [00:39<00:16, 6917.40it/s] 72%|███████▏  | 287187/400000 [00:40<00:16, 6952.27it/s] 72%|███████▏  | 287884/400000 [00:40<00:16, 6823.36it/s] 72%|███████▏  | 288571/400000 [00:40<00:16, 6836.28it/s] 72%|███████▏  | 289264/400000 [00:40<00:16, 6863.53it/s] 72%|███████▏  | 289992/400000 [00:40<00:15, 6982.18it/s] 73%|███████▎  | 290766/400000 [00:40<00:15, 7193.00it/s] 73%|███████▎  | 291491/400000 [00:40<00:15, 7208.83it/s] 73%|███████▎  | 292291/400000 [00:40<00:14, 7427.02it/s] 73%|███████▎  | 293037/400000 [00:40<00:15, 7123.07it/s] 73%|███████▎  | 293754/400000 [00:40<00:16, 6569.92it/s] 74%|███████▎  | 294423/400000 [00:41<00:16, 6501.67it/s] 74%|███████▍  | 295160/400000 [00:41<00:15, 6738.32it/s] 74%|███████▍  | 295914/400000 [00:41<00:14, 6958.69it/s] 74%|███████▍  | 296618/400000 [00:41<00:15, 6821.75it/s] 74%|███████▍  | 297310/400000 [00:41<00:14, 6850.84it/s] 75%|███████▍  | 298028/400000 [00:41<00:14, 6943.15it/s] 75%|███████▍  | 298774/400000 [00:41<00:14, 7089.73it/s] 75%|███████▍  | 299524/400000 [00:41<00:13, 7207.40it/s] 75%|███████▌  | 300248/400000 [00:41<00:14, 7102.07it/s] 75%|███████▌  | 300961/400000 [00:42<00:14, 7029.39it/s] 75%|███████▌  | 301666/400000 [00:42<00:14, 6990.35it/s] 76%|███████▌  | 302367/400000 [00:42<00:14, 6969.06it/s] 76%|███████▌  | 303065/400000 [00:42<00:14, 6895.53it/s] 76%|███████▌  | 303756/400000 [00:42<00:14, 6777.42it/s] 76%|███████▌  | 304437/400000 [00:42<00:14, 6786.12it/s] 76%|███████▋  | 305148/400000 [00:42<00:13, 6877.84it/s] 76%|███████▋  | 305890/400000 [00:42<00:13, 7030.01it/s] 77%|███████▋  | 306630/400000 [00:42<00:13, 7135.05it/s] 77%|███████▋  | 307345/400000 [00:42<00:12, 7137.25it/s] 77%|███████▋  | 308097/400000 [00:43<00:12, 7245.94it/s] 77%|███████▋  | 308823/400000 [00:43<00:12, 7141.38it/s] 77%|███████▋  | 309539/400000 [00:43<00:12, 7049.73it/s] 78%|███████▊  | 310245/400000 [00:43<00:12, 7048.32it/s] 78%|███████▊  | 310951/400000 [00:43<00:12, 6890.46it/s] 78%|███████▊  | 311671/400000 [00:43<00:12, 6976.63it/s] 78%|███████▊  | 312370/400000 [00:43<00:12, 6959.84it/s] 78%|███████▊  | 313076/400000 [00:43<00:12, 6989.45it/s] 78%|███████▊  | 313832/400000 [00:43<00:12, 7147.19it/s] 79%|███████▊  | 314549/400000 [00:43<00:12, 7009.31it/s] 79%|███████▉  | 315252/400000 [00:44<00:12, 6980.19it/s] 79%|███████▉  | 316002/400000 [00:44<00:11, 7127.98it/s] 79%|███████▉  | 316725/400000 [00:44<00:11, 7157.04it/s] 79%|███████▉  | 317442/400000 [00:44<00:12, 6632.06it/s] 80%|███████▉  | 318177/400000 [00:44<00:11, 6830.62it/s] 80%|███████▉  | 318977/400000 [00:44<00:11, 7142.09it/s] 80%|███████▉  | 319752/400000 [00:44<00:10, 7311.49it/s] 80%|████████  | 320547/400000 [00:44<00:10, 7491.01it/s] 80%|████████  | 321345/400000 [00:44<00:10, 7629.47it/s] 81%|████████  | 322113/400000 [00:44<00:10, 7545.62it/s] 81%|████████  | 322872/400000 [00:45<00:10, 7514.45it/s] 81%|████████  | 323627/400000 [00:45<00:10, 7483.69it/s] 81%|████████  | 324378/400000 [00:45<00:10, 7127.13it/s] 81%|████████▏ | 325096/400000 [00:45<00:10, 7139.84it/s] 81%|████████▏ | 325838/400000 [00:45<00:10, 7218.33it/s] 82%|████████▏ | 326598/400000 [00:45<00:10, 7328.45it/s] 82%|████████▏ | 327334/400000 [00:45<00:09, 7271.64it/s] 82%|████████▏ | 328063/400000 [00:45<00:10, 7191.97it/s] 82%|████████▏ | 328784/400000 [00:45<00:10, 7049.37it/s] 82%|████████▏ | 329503/400000 [00:46<00:09, 7088.98it/s] 83%|████████▎ | 330246/400000 [00:46<00:09, 7186.83it/s] 83%|████████▎ | 330986/400000 [00:46<00:09, 7247.22it/s] 83%|████████▎ | 331741/400000 [00:46<00:09, 7332.17it/s] 83%|████████▎ | 332497/400000 [00:46<00:09, 7398.80it/s] 83%|████████▎ | 333238/400000 [00:46<00:09, 7211.49it/s] 83%|████████▎ | 333961/400000 [00:46<00:09, 7030.10it/s] 84%|████████▎ | 334667/400000 [00:46<00:09, 6832.19it/s] 84%|████████▍ | 335353/400000 [00:46<00:09, 6808.26it/s] 84%|████████▍ | 336054/400000 [00:46<00:09, 6865.44it/s] 84%|████████▍ | 336816/400000 [00:47<00:08, 7075.35it/s] 84%|████████▍ | 337540/400000 [00:47<00:08, 7123.35it/s] 85%|████████▍ | 338255/400000 [00:47<00:08, 7038.05it/s] 85%|████████▍ | 338961/400000 [00:47<00:08, 6843.60it/s] 85%|████████▍ | 339648/400000 [00:47<00:08, 6770.36it/s] 85%|████████▌ | 340406/400000 [00:47<00:08, 6994.18it/s] 85%|████████▌ | 341123/400000 [00:47<00:08, 7043.19it/s] 85%|████████▌ | 341851/400000 [00:47<00:08, 7112.48it/s] 86%|████████▌ | 342602/400000 [00:47<00:07, 7225.16it/s] 86%|████████▌ | 343335/400000 [00:47<00:07, 7254.21it/s] 86%|████████▌ | 344062/400000 [00:48<00:07, 7248.87it/s] 86%|████████▌ | 344788/400000 [00:48<00:07, 7161.44it/s] 86%|████████▋ | 345505/400000 [00:48<00:07, 7095.97it/s] 87%|████████▋ | 346216/400000 [00:48<00:07, 6980.71it/s] 87%|████████▋ | 346915/400000 [00:48<00:07, 6802.66it/s] 87%|████████▋ | 347597/400000 [00:48<00:07, 6745.03it/s] 87%|████████▋ | 348284/400000 [00:48<00:07, 6781.31it/s] 87%|████████▋ | 348983/400000 [00:48<00:07, 6841.98it/s] 87%|████████▋ | 349691/400000 [00:48<00:07, 6911.39it/s] 88%|████████▊ | 350411/400000 [00:48<00:07, 6992.28it/s] 88%|████████▊ | 351189/400000 [00:49<00:06, 7210.93it/s] 88%|████████▊ | 351921/400000 [00:49<00:06, 7241.94it/s] 88%|████████▊ | 352673/400000 [00:49<00:06, 7321.79it/s] 88%|████████▊ | 353407/400000 [00:49<00:06, 7123.14it/s] 89%|████████▊ | 354122/400000 [00:49<00:06, 6901.96it/s] 89%|████████▊ | 354888/400000 [00:49<00:06, 7111.94it/s] 89%|████████▉ | 355660/400000 [00:49<00:06, 7283.64it/s] 89%|████████▉ | 356414/400000 [00:49<00:05, 7358.49it/s] 89%|████████▉ | 357153/400000 [00:49<00:05, 7289.25it/s] 89%|████████▉ | 357884/400000 [00:50<00:05, 7117.40it/s] 90%|████████▉ | 358619/400000 [00:50<00:05, 7182.14it/s] 90%|████████▉ | 359339/400000 [00:50<00:05, 7024.33it/s] 90%|█████████ | 360059/400000 [00:50<00:05, 7073.21it/s] 90%|█████████ | 360783/400000 [00:50<00:05, 7121.55it/s] 90%|█████████ | 361541/400000 [00:50<00:05, 7252.79it/s] 91%|█████████ | 362303/400000 [00:50<00:05, 7357.32it/s] 91%|█████████ | 363041/400000 [00:50<00:05, 7341.10it/s] 91%|█████████ | 363813/400000 [00:50<00:04, 7450.72it/s] 91%|█████████ | 364560/400000 [00:50<00:04, 7380.52it/s] 91%|█████████▏| 365299/400000 [00:51<00:04, 7352.75it/s] 92%|█████████▏| 366053/400000 [00:51<00:04, 7407.70it/s] 92%|█████████▏| 366818/400000 [00:51<00:04, 7478.42it/s] 92%|█████████▏| 367583/400000 [00:51<00:04, 7527.96it/s] 92%|█████████▏| 368351/400000 [00:51<00:04, 7572.47it/s] 92%|█████████▏| 369109/400000 [00:51<00:04, 7451.62it/s] 92%|█████████▏| 369885/400000 [00:51<00:03, 7539.57it/s] 93%|█████████▎| 370647/400000 [00:51<00:03, 7561.64it/s] 93%|█████████▎| 371449/400000 [00:51<00:03, 7691.04it/s] 93%|█████████▎| 372225/400000 [00:51<00:03, 7711.13it/s] 93%|█████████▎| 372997/400000 [00:52<00:03, 7620.00it/s] 93%|█████████▎| 373760/400000 [00:52<00:03, 7604.80it/s] 94%|█████████▎| 374536/400000 [00:52<00:03, 7649.23it/s] 94%|█████████▍| 375304/400000 [00:52<00:03, 7657.71it/s] 94%|█████████▍| 376100/400000 [00:52<00:03, 7743.90it/s] 94%|█████████▍| 376875/400000 [00:52<00:03, 7612.45it/s] 94%|█████████▍| 377638/400000 [00:52<00:02, 7484.17it/s] 95%|█████████▍| 378388/400000 [00:52<00:02, 7400.43it/s] 95%|█████████▍| 379141/400000 [00:52<00:02, 7437.53it/s] 95%|█████████▍| 379900/400000 [00:52<00:02, 7481.98it/s] 95%|█████████▌| 380654/400000 [00:53<00:02, 7498.25it/s] 95%|█████████▌| 381405/400000 [00:53<00:02, 7403.51it/s] 96%|█████████▌| 382166/400000 [00:53<00:02, 7463.72it/s] 96%|█████████▌| 382933/400000 [00:53<00:02, 7524.04it/s] 96%|█████████▌| 383690/400000 [00:53<00:02, 7537.50it/s] 96%|█████████▌| 384445/400000 [00:53<00:02, 7415.86it/s] 96%|█████████▋| 385216/400000 [00:53<00:01, 7500.04it/s] 96%|█████████▋| 385967/400000 [00:53<00:01, 7440.59it/s] 97%|█████████▋| 386754/400000 [00:53<00:01, 7564.21it/s] 97%|█████████▋| 387513/400000 [00:53<00:01, 7569.57it/s] 97%|█████████▋| 388271/400000 [00:54<00:01, 7557.27it/s] 97%|█████████▋| 389045/400000 [00:54<00:01, 7608.54it/s] 97%|█████████▋| 389807/400000 [00:54<00:01, 7477.06it/s] 98%|█████████▊| 390556/400000 [00:54<00:01, 7468.30it/s] 98%|█████████▊| 391304/400000 [00:54<00:01, 7460.11it/s] 98%|█████████▊| 392051/400000 [00:54<00:01, 7412.47it/s] 98%|█████████▊| 392793/400000 [00:54<00:00, 7355.30it/s] 98%|█████████▊| 393529/400000 [00:54<00:00, 7331.57it/s] 99%|█████████▊| 394270/400000 [00:54<00:00, 7354.87it/s] 99%|█████████▉| 395013/400000 [00:54<00:00, 7374.77it/s] 99%|█████████▉| 395751/400000 [00:55<00:00, 7181.20it/s] 99%|█████████▉| 396474/400000 [00:55<00:00, 7193.44it/s] 99%|█████████▉| 397195/400000 [00:55<00:00, 7179.10it/s] 99%|█████████▉| 397914/400000 [00:55<00:00, 7168.27it/s]100%|█████████▉| 398632/400000 [00:55<00:00, 7104.28it/s]100%|█████████▉| 399343/400000 [00:55<00:00, 7095.36it/s]100%|█████████▉| 399999/400000 [00:55<00:00, 7182.83it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f250af5e0f0>, <torchtext.data.dataset.TabularDataset object at 0x7f250af5e240>, <torchtext.vocab.Vocab object at 0x7f250af5e160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 23.01 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 43.76 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 21.83 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 21.83 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.03 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.03 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.84 file/s]2020-06-06 18:19:25.912916: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-06 18:19:25.917765: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-06 18:19:25.917991: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56529cd16ea0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-06 18:19:25.918009: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'test', 'mnist2', 'cifar10', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:05, 150694.33it/s] 70%|██████▉   | 6897664/9912422 [00:00<00:14, 215075.73it/s]9920512it [00:00, 41657465.93it/s]                           
0it [00:00, ?it/s]32768it [00:00, 483573.89it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 156354.40it/s]1654784it [00:00, 10904477.40it/s]                         
0it [00:00, ?it/s]8192it [00:00, 162026.09it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
