## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-09-30-12-25_97277fff3460fefb27a1009c960a6fe4c8eb6311.py


### Error 1, [Traceback at line 163](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-09-30-12-25_97277fff3460fefb27a1009c960a6fe4c8eb6311.py#L163)<br />163..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 490, in load_entry_point
<br />    return get_distribution(dist).load_entry_point(group, name)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2854, in load_entry_point
<br />    return ep.load()
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2445, in load
<br />    return self.resolve()
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2451, in resolve
<br />    module = __import__(self.module_name, fromlist=['__name__'], level=0)
<br />  File "https://github.com/arita37/mlmodels/tree/97277fff3460fefb27a1009c960a6fe4c8eb6311/mlmodels/optim.py", line 32, in <module>
<br />    from mlmodels.models import model_create, module_load
<br />  File "https://github.com/arita37/mlmodels/tree/97277fff3460fefb27a1009c960a6fe4c8eb6311/mlmodels/models.py", line 92
<br />    else :
<br />         ^
<br />IndentationError: unindent does not match any outer indentation level



### Error 2, [Traceback at line 195](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-09-30-12-25_97277fff3460fefb27a1009c960a6fe4c8eb6311.py#L195)<br />195..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/97277fff3460fefb27a1009c960a6fe4c8eb6311/mlmodels/optim.py", line 32, in <module>
<br />    from mlmodels.models import model_create, module_load
<br />  File "https://github.com/arita37/mlmodels/tree/97277fff3460fefb27a1009c960a6fe4c8eb6311/mlmodels/models.py", line 92
<br />    else :
<br />         ^
<br />IndentationError: unindent does not match any outer indentation level



### Error 3, [Traceback at line 207](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-09-30-12-25_97277fff3460fefb27a1009c960a6fe4c8eb6311.py#L207)<br />207..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/97277fff3460fefb27a1009c960a6fe4c8eb6311/mlmodels/model_keras/textcnn.py", line 31, in <module>
<br />    from mlmodels.dataloader import DataLoader
<br />  File "https://github.com/arita37/mlmodels/tree/97277fff3460fefb27a1009c960a6fe4c8eb6311/mlmodels/dataloader.py", line 318
<br />    else :
<br />         ^
<br />IndentationError: unindent does not match any outer indentation level



### Error 4, [Traceback at line 214](https://github.com/arita37/mlmodels_store/blob/master/log_pullrequest/log_pr_2020-09-30-12-25_97277fff3460fefb27a1009c960a6fe4c8eb6311.py#L214)<br />214..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/bin/ml_test", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_test')()
<br />  File "https://github.com/arita37/mlmodels/tree/97277fff3460fefb27a1009c960a6fe4c8eb6311/mlmodels/ztest.py", line 655, in main
<br />    globals()[arg.do](arg)
<br />  File "https://github.com/arita37/mlmodels/tree/97277fff3460fefb27a1009c960a6fe4c8eb6311/mlmodels/ztest.py", line 424, in test_pullrequest
<br />    raise Exception(f"Unknown dataset type", x)
<br />Exception: ('Unknown dataset type', 'IndentationError: unindent does not match any outer indentation level\n')
