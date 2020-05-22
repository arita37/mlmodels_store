## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py


### Error 1, [Traceback at line 46](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L46)<br />46..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 405, in preprocess
<br />    nb, resources = super(ExecutePreprocessor, self).preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 69, in preprocess
<br />    nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 450, in preprocess_cell
<br />    raise CellExecutionError.from_cell_and_msg(cell, reply['content'])
<br />nbconvert.preprocessors.execute.CellExecutionError: An error occurred while executing the following cell:
<br />------------------
<br />#### Install MLMODELS in Colab
<br />%%capture
<br />!  bash <(wget -qO- https://cutt.ly/mlmodels)
<br />!pip install pydantic==1.4 --force
<br />
<br />### Restart Runtime AFTER Install
<br />import os, time
<br />os.kill(os.getpid(), 9)
<br />------------------
<br />
<br />
<br />UsageError: Line magic function `%%capture` not found.



### Error 2, [Traceback at line 93](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L93)<br />93..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 403, in preprocess
<br />    with self.setup_preprocessor(nb, resources, km=km):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/contextlib.py", line 81, in __enter__
<br />    return next(self.gen)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 345, in setup_preprocessor
<br />    self.km, self.kc = self.start_new_kernel(**kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 291, in start_new_kernel
<br />    km.start_kernel(extra_arguments=self.extra_arguments, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 301, in start_kernel
<br />    kernel_cmd, kw = self.pre_start_kernel(**kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 254, in pre_start_kernel
<br />    kernel_cmd = self.format_kernel_cmd(extra_arguments=extra_arguments)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 178, in format_kernel_cmd
<br />    cmd = self.kernel_spec.argv + extra_arguments
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 84, in kernel_spec
<br />    self._kernel_spec = self.kernel_spec_manager.get_kernel_spec(self.kernel_name)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/kernelspec.py", line 235, in get_kernel_spec
<br />    raise NoSuchKernel(kernel_name)
<br />jupyter_client.kernelspec.NoSuchKernel: No such kernel named mlmodels
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vison_fashion_MNIST.ipynb 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//vison_fashion_MNIST.ipynb to html
<br />[NbConvertApp] Executing notebook with kernel: python3
<br />[NbConvertApp] ERROR | Error while converting 'https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//vison_fashion_MNIST.ipynb'



### Error 3, [Traceback at line 152](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L152)<br />152..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 405, in preprocess
<br />    nb, resources = super(ExecutePreprocessor, self).preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 69, in preprocess
<br />    nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 448, in preprocess_cell
<br />    raise CellExecutionError.from_cell_and_msg(cell, out)
<br />nbconvert.preprocessors.execute.CellExecutionError: An error occurred while executing the following cell:
<br />------------------
<br />from google.colab import drive
<br />drive.mount('/content/drive')
<br />------------------
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 4, [Traceback at line 182](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L182)<br />182..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m<ipython-input-1-d5df0069828e>[0m in [0;36m<module>[0;34m[0m
<br />[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'google.colab'
<br />ModuleNotFoundError: No module named 'google.colab'



### Error 5, [Traceback at line 201](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L201)<br />201..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 405, in preprocess
<br />    nb, resources = super(ExecutePreprocessor, self).preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 69, in preprocess
<br />    nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 448, in preprocess_cell
<br />    raise CellExecutionError.from_cell_and_msg(cell, out)
<br />nbconvert.preprocessors.execute.CellExecutionError: An error occurred while executing the following cell:
<br />------------------
<br />from google.colab import drive
<br />drive.mount('/content/drive')
<br />------------------
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 6, [Traceback at line 231](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L231)<br />231..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m<ipython-input-1-d5df0069828e>[0m in [0;36m<module>[0;34m[0m
<br />[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'google.colab'
<br />ModuleNotFoundError: No module named 'google.colab'



### Error 7, [Traceback at line 250](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L250)<br />250..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 405, in preprocess
<br />    nb, resources = super(ExecutePreprocessor, self).preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 69, in preprocess
<br />    nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 448, in preprocess_cell
<br />    raise CellExecutionError.from_cell_and_msg(cell, out)
<br />nbconvert.preprocessors.execute.CellExecutionError: An error occurred while executing the following cell:
<br />------------------
<br />from google.colab import drive
<br />drive.mount('/content/drive')
<br />------------------
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 8, [Traceback at line 280](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L280)<br />280..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m<ipython-input-1-d5df0069828e>[0m in [0;36m<module>[0;34m[0m
<br />[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'google.colab'
<br />ModuleNotFoundError: No module named 'google.colab'



### Error 9, [Traceback at line 297](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L297)<br />297..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 403, in preprocess
<br />    with self.setup_preprocessor(nb, resources, km=km):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/contextlib.py", line 81, in __enter__
<br />    return next(self.gen)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 345, in setup_preprocessor
<br />    self.km, self.kc = self.start_new_kernel(**kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 291, in start_new_kernel
<br />    km.start_kernel(extra_arguments=self.extra_arguments, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 301, in start_kernel
<br />    kernel_cmd, kw = self.pre_start_kernel(**kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 254, in pre_start_kernel
<br />    kernel_cmd = self.format_kernel_cmd(extra_arguments=extra_arguments)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 178, in format_kernel_cmd
<br />    cmd = self.kernel_spec.argv + extra_arguments
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 84, in kernel_spec
<br />    self._kernel_spec = self.kernel_spec_manager.get_kernel_spec(self.kernel_name)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/kernelspec.py", line 235, in get_kernel_spec
<br />    raise NoSuchKernel(kernel_name)
<br />jupyter_client.kernelspec.NoSuchKernel: No such kernel named conda-env-mlmodels2-py
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn_titanic_randomForest.ipynb 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//sklearn_titanic_randomForest.ipynb to html
<br />[NbConvertApp] Executing notebook with kernel: python3
<br />[NbConvertApp] ERROR | Error while converting 'https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//sklearn_titanic_randomForest.ipynb'



### Error 10, [Traceback at line 356](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L356)<br />356..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 405, in preprocess
<br />    nb, resources = super(ExecutePreprocessor, self).preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 69, in preprocess
<br />    nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 448, in preprocess_cell
<br />    raise CellExecutionError.from_cell_and_msg(cell, out)
<br />nbconvert.preprocessors.execute.CellExecutionError: An error occurred while executing the following cell:
<br />------------------
<br />from mlmodels.models import module_load
<br />
<br />model_uri    = "model_sklearn.sklearn.py"
<br />module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />
<br />model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
<br />    'choice':'json',
<br />    'config_mode':'test',
<br />    'data_path':'../mlmodels/dataset/json/sklearn_titanic_randomForest.json'
<br />})
<br />------------------
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 11, [Traceback at line 394](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L394)<br />394..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
<br />[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m    127[0m [0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'mlmodels.model_sklearn.sklearn'
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 12, [Traceback at line 415](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L415)<br />415..[0;31mIndexError[0m                                Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mIndexError[0m: tuple index out of range
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 13, [Traceback at line 425](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L425)<br />425..[0;31mNameError[0m                                 Traceback (most recent call last)
<br />[0;32m<ipython-input-2-41f4efe8d631>[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      2[0m [0;34m[0m[0m
<br />[1;32m      3[0m [0mmodel_uri[0m    [0;34m=[0m [0;34m"model_sklearn.sklearn.py"[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m----> 4[0;31m [0mmodule[0m        [0;34m=[0m  [0mmodule_load[0m[0;34m([0m [0mmodel_uri[0m[0;34m=[0m [0mmodel_uri[0m [0;34m)[0m                           [0;31m# Load file definition[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      5[0m [0;34m[0m[0m
<br />[1;32m      6[0m model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
<br />
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     87[0m [0;34m[0m[0m
<br />[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     90[0m [0;34m[0m[0m
<br />[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 14, [Traceback at line 454](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L454)<br />454..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 405, in preprocess
<br />    nb, resources = super(ExecutePreprocessor, self).preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 69, in preprocess
<br />    nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 448, in preprocess_cell
<br />    raise CellExecutionError.from_cell_and_msg(cell, out)
<br />nbconvert.preprocessors.execute.CellExecutionError: An error occurred while executing the following cell:
<br />------------------
<br />from mlmodels.models import module_load
<br />from mlmodels.optim import optim
<br />import json
<br />data_path = '../mlmodels/dataset/json/hyper_titanic_randomForest.json'  
<br />pars = json.load(open( data_path , mode='r'))
<br />for key, pdict in  pars.items() :
<br />  globals()[key] = pdict   
<br />
<br />hypermodel_pars = test['hypermodel_pars']
<br />model_pars      = test['model_pars']
<br />data_pars       = test['data_pars']
<br />compute_pars    = test['compute_pars']
<br />out_pars        = test['out_pars']
<br />
<br />model_uri    = "model_sklearn.sklearn.py"
<br />module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />
<br />model_pars_update = optim(
<br />    model_uri       = model_uri,
<br />    hypermodel_pars = hypermodel_pars,
<br />    model_pars      = model_pars,
<br />    data_pars       = data_pars,
<br />    compute_pars    = compute_pars,
<br />    out_pars        = out_pars
<br />)
<br />------------------
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 15, [Traceback at line 507](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L507)<br />507..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m<ipython-input-2-4bbd5c2b0e99>[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      3[0m [0;32mimport[0m [0mjson[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      4[0m [0mdata_path[0m [0;34m=[0m [0;34m'../mlmodels/dataset/json/hyper_titanic_randomForest.json'[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m----> 5[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      6[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      7[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '../mlmodels/dataset/json/hyper_titanic_randomForest.json'
<br />FileNotFoundError: [Errno 2] No such file or directory: '../mlmodels/dataset/json/hyper_titanic_randomForest.json'



### Error 16, [Traceback at line 527](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L527)<br />527..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 403, in preprocess
<br />    with self.setup_preprocessor(nb, resources, km=km):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/contextlib.py", line 81, in __enter__
<br />    return next(self.gen)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 345, in setup_preprocessor
<br />    self.km, self.kc = self.start_new_kernel(**kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 291, in start_new_kernel
<br />    km.start_kernel(extra_arguments=self.extra_arguments, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 301, in start_kernel
<br />    kernel_cmd, kw = self.pre_start_kernel(**kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 254, in pre_start_kernel
<br />    kernel_cmd = self.format_kernel_cmd(extra_arguments=extra_arguments)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 178, in format_kernel_cmd
<br />    cmd = self.kernel_spec.argv + extra_arguments
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 84, in kernel_spec
<br />    self._kernel_spec = self.kernel_spec_manager.get_kernel_spec(self.kernel_name)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/kernelspec.py", line 235, in get_kernel_spec
<br />    raise NoSuchKernel(kernel_name)
<br />jupyter_client.kernelspec.NoSuchKernel: No such kernel named conda-env-mlmodels2-py
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_titanic.ipynb 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//lightgbm_titanic.ipynb to html



### Error 17, [Traceback at line 584](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L584)<br />584..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 403, in preprocess
<br />    with self.setup_preprocessor(nb, resources, km=km):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/contextlib.py", line 81, in __enter__
<br />    return next(self.gen)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 345, in setup_preprocessor
<br />    self.km, self.kc = self.start_new_kernel(**kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 291, in start_new_kernel
<br />    km.start_kernel(extra_arguments=self.extra_arguments, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 301, in start_kernel
<br />    kernel_cmd, kw = self.pre_start_kernel(**kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 254, in pre_start_kernel
<br />    kernel_cmd = self.format_kernel_cmd(extra_arguments=extra_arguments)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 178, in format_kernel_cmd
<br />    cmd = self.kernel_spec.argv + extra_arguments
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 84, in kernel_spec
<br />    self._kernel_spec = self.kernel_spec_manager.get_kernel_spec(self.kernel_name)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/kernelspec.py", line 235, in get_kernel_spec
<br />    raise NoSuchKernel(kernel_name)
<br />jupyter_client.kernelspec.NoSuchKernel: No such kernel named mlmodels
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//mnist_mlmodels_.ipynb 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//mnist_mlmodels_.ipynb to html
<br />[NbConvertApp] Executing notebook with kernel: python3
<br />[NbConvertApp] ERROR | Error while converting 'https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//mnist_mlmodels_.ipynb'



### Error 18, [Traceback at line 643](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L643)<br />643..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 405, in preprocess
<br />    nb, resources = super(ExecutePreprocessor, self).preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 69, in preprocess
<br />    nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 448, in preprocess_cell
<br />    raise CellExecutionError.from_cell_and_msg(cell, out)
<br />nbconvert.preprocessors.execute.CellExecutionError: An error occurred while executing the following cell:
<br />------------------
<br />from google.colab import drive
<br />drive.mount('/content/drive')
<br />------------------
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 19, [Traceback at line 673](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L673)<br />673..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m<ipython-input-1-d5df0069828e>[0m in [0;36m<module>[0;34m[0m
<br />[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'google.colab'
<br />ModuleNotFoundError: No module named 'google.colab'



### Error 20, [Traceback at line 692](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L692)<br />692..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 405, in preprocess
<br />    nb, resources = super(ExecutePreprocessor, self).preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 69, in preprocess
<br />    nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 448, in preprocess_cell
<br />    raise CellExecutionError.from_cell_and_msg(cell, out)
<br />nbconvert.preprocessors.execute.CellExecutionError: An error occurred while executing the following cell:
<br />------------------
<br />from mlmodels.models import module_load
<br />from mlmodels.util import load_config
<br />
<br />model_uri    = "model_sklearn.sklearn.py"
<br />module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />
<br />model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
<br />    'choice':'json',
<br />    'config_mode':'test',
<br />    'data_path':'../mlmodels/dataset/json/sklearn_titanic_svm.json'
<br />})
<br />------------------
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 21, [Traceback at line 731](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L731)<br />731..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
<br />[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m    127[0m [0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'mlmodels.model_sklearn.sklearn'
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 22, [Traceback at line 752](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L752)<br />752..[0;31mIndexError[0m                                Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mIndexError[0m: tuple index out of range
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 23, [Traceback at line 762](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L762)<br />762..[0;31mNameError[0m                                 Traceback (most recent call last)
<br />[0;32m<ipython-input-2-779dc93e4a41>[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      3[0m [0;34m[0m[0m
<br />[1;32m      4[0m [0mmodel_uri[0m    [0;34m=[0m [0;34m"model_sklearn.sklearn.py"[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m----> 5[0;31m [0mmodule[0m        [0;34m=[0m  [0mmodule_load[0m[0;34m([0m [0mmodel_uri[0m[0;34m=[0m [0mmodel_uri[0m [0;34m)[0m                           [0;31m# Load file definition[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      6[0m [0;34m[0m[0m
<br />[1;32m      7[0m model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
<br />
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     87[0m [0;34m[0m[0m
<br />[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     90[0m [0;34m[0m[0m
<br />[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 24, [Traceback at line 789](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L789)<br />789..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 403, in preprocess
<br />    with self.setup_preprocessor(nb, resources, km=km):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/contextlib.py", line 81, in __enter__
<br />    return next(self.gen)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 345, in setup_preprocessor
<br />    self.km, self.kc = self.start_new_kernel(**kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 291, in start_new_kernel
<br />    km.start_kernel(extra_arguments=self.extra_arguments, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 301, in start_kernel
<br />    kernel_cmd, kw = self.pre_start_kernel(**kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 254, in pre_start_kernel
<br />    kernel_cmd = self.format_kernel_cmd(extra_arguments=extra_arguments)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 178, in format_kernel_cmd
<br />    cmd = self.kernel_spec.argv + extra_arguments
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 84, in kernel_spec
<br />    self._kernel_spec = self.kernel_spec_manager.get_kernel_spec(self.kernel_name)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/kernelspec.py", line 235, in get_kernel_spec
<br />    raise NoSuchKernel(kernel_name)
<br />jupyter_client.kernelspec.NoSuchKernel: No such kernel named mlmodels
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_glass.ipynb 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//lightgbm_glass.ipynb to html



### Error 25, [Traceback at line 846](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L846)<br />846..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 403, in preprocess
<br />    with self.setup_preprocessor(nb, resources, km=km):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/contextlib.py", line 81, in __enter__
<br />    return next(self.gen)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 345, in setup_preprocessor
<br />    self.km, self.kc = self.start_new_kernel(**kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 291, in start_new_kernel
<br />    km.start_kernel(extra_arguments=self.extra_arguments, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 301, in start_kernel
<br />    kernel_cmd, kw = self.pre_start_kernel(**kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 254, in pre_start_kernel
<br />    kernel_cmd = self.format_kernel_cmd(extra_arguments=extra_arguments)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 178, in format_kernel_cmd
<br />    cmd = self.kernel_spec.argv + extra_arguments
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 84, in kernel_spec
<br />    self._kernel_spec = self.kernel_spec_manager.get_kernel_spec(self.kernel_name)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/kernelspec.py", line 235, in get_kernel_spec
<br />    raise NoSuchKernel(kernel_name)
<br />jupyter_client.kernelspec.NoSuchKernel: No such kernel named conda-env-mlmodels2-py
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn.ipynb 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//sklearn.ipynb to html
<br />[NbConvertApp] Executing notebook with kernel: python3
<br />[NbConvertApp] ERROR | Error while converting 'https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//sklearn.ipynb'



### Error 26, [Traceback at line 905](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L905)<br />905..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 405, in preprocess
<br />    nb, resources = super(ExecutePreprocessor, self).preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 69, in preprocess
<br />    nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 448, in preprocess_cell
<br />    raise CellExecutionError.from_cell_and_msg(cell, out)
<br />nbconvert.preprocessors.execute.CellExecutionError: An error occurred while executing the following cell:
<br />------------------
<br />from mlmodels.models import module_load
<br />
<br />module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
<br />model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
<br />------------------
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 27, [Traceback at line 938](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L938)<br />938..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
<br />[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m    127[0m [0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'mlmodels.model_sklearn.sklearn'
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 28, [Traceback at line 959](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L959)<br />959..[0;31mIndexError[0m                                Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mIndexError[0m: tuple index out of range
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 29, [Traceback at line 969](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L969)<br />969..[0;31mNameError[0m                                 Traceback (most recent call last)
<br />[0;32m<ipython-input-3-6f4d82b0f29f>[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      1[0m [0;32mfrom[0m [0mmlmodels[0m[0;34m.[0m[0mmodels[0m [0;32mimport[0m [0mmodule_load[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      2[0m [0;34m[0m[0m
<br />[0;32m----> 3[0;31m [0mmodule[0m        [0;34m=[0m  [0mmodule_load[0m[0;34m([0m [0mmodel_uri[0m[0;34m=[0m [0mmodel_uri[0m [0;34m)[0m                           [0;31m# Load file definition[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      4[0m [0mmodel[0m         [0;34m=[0m  [0mmodule[0m[0;34m.[0m[0mModel[0m[0;34m([0m[0mmodel_pars[0m[0;34m=[0m[0mmodel_pars[0m[0;34m,[0m [0mdata_pars[0m[0;34m=[0m[0mdata_pars[0m[0;34m,[0m [0mcompute_pars[0m[0;34m=[0m[0mcompute_pars[0m[0;34m)[0m             [0;31m# Create Model instance[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      5[0m [0mmodel[0m[0;34m,[0m [0msess[0m   [0;34m=[0m  [0mmodule[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mdata_pars[0m[0;34m=[0m[0mdata_pars[0m[0;34m,[0m [0mcompute_pars[0m[0;34m=[0m[0mcompute_pars[0m[0;34m,[0m [0mout_pars[0m[0;34m=[0m[0mout_pars[0m[0;34m)[0m          [0;31m# fit the model[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     87[0m [0;34m[0m[0m
<br />[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     90[0m [0;34m[0m[0m
<br />[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 30, [Traceback at line 998](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L998)<br />998..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 405, in preprocess
<br />    nb, resources = super(ExecutePreprocessor, self).preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 69, in preprocess
<br />    nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 448, in preprocess_cell
<br />    raise CellExecutionError.from_cell_and_msg(cell, out)
<br />nbconvert.preprocessors.execute.CellExecutionError: An error occurred while executing the following cell:
<br />------------------
<br />from mlmodels.models import module_load
<br />from mlmodels.util import load_config
<br />
<br />model_uri    = "model_gluon.gluon_automl.py"
<br />module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />
<br />model_pars, data_pars, compute_pars, out_pars = module.get_params(
<br />    choice='json',
<br />    config_mode= 'test',
<br />    data_path= '../mlmodels/dataset/json/gluon_automl.json'
<br />)
<br />------------------
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 31, [Traceback at line 1037](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1037)<br />1037..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m<ipython-input-2-f529147a19d9>[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      8[0m     [0mchoice[0m[0;34m=[0m[0;34m'json'[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      9[0m     [0mconfig_mode[0m[0;34m=[0m [0;34m'test'[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 10[0;31m     [0mdata_path[0m[0;34m=[0m [0;34m'../mlmodels/dataset/json/gluon_automl.json'[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     11[0m )
<br />
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/model_gluon/gluon_automl.py[0m in [0;36mget_params[0;34m(choice, data_path, config_mode, **kw)[0m
<br />[1;32m     80[0m             __file__)).parent.parent / "model_gluon/gluon_automl.json" if data_path == "dataset/" else data_path
<br />[1;32m     81[0m [0;34m[0m[0m
<br />[0;32m---> 82[0;31m         [0;32mwith[0m [0mopen[0m[0;34m([0m[0mdata_path[0m[0;34m,[0m [0mencoding[0m[0;34m=[0m[0;34m'utf-8'[0m[0;34m)[0m [0;32mas[0m [0mconfig_f[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     83[0m             [0mconfig[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mconfig_f[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m     84[0m             [0mconfig[0m [0;34m=[0m [0mconfig[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '../mlmodels/dataset/json/gluon_automl.json'
<br />FileNotFoundError: [Errno 2] No such file or directory: '../mlmodels/dataset/json/gluon_automl.json'



### Error 32, [Traceback at line 1063](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1063)<br />1063..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 403, in preprocess
<br />    with self.setup_preprocessor(nb, resources, km=km):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/contextlib.py", line 81, in __enter__
<br />    return next(self.gen)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 345, in setup_preprocessor
<br />    self.km, self.kc = self.start_new_kernel(**kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 291, in start_new_kernel
<br />    km.start_kernel(extra_arguments=self.extra_arguments, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 301, in start_kernel
<br />    kernel_cmd, kw = self.pre_start_kernel(**kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 254, in pre_start_kernel
<br />    kernel_cmd = self.format_kernel_cmd(extra_arguments=extra_arguments)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 178, in format_kernel_cmd
<br />    cmd = self.kernel_spec.argv + extra_arguments
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/manager.py", line 84, in kernel_spec
<br />    self._kernel_spec = self.kernel_spec_manager.get_kernel_spec(self.kernel_name)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_client/kernelspec.py", line 235, in get_kernel_spec
<br />    raise NoSuchKernel(kernel_name)
<br />jupyter_client.kernelspec.NoSuchKernel: No such kernel named myenv
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//keras-textcnn.ipynb 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//keras-textcnn.ipynb to html
<br />[NbConvertApp] Executing notebook with kernel: python3
<br />2020-05-22 09:10:25.280870: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
<br />2020-05-22 09:10:25.284993: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
<br />2020-05-22 09:10:25.285169: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f88184eb60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
<br />2020-05-22 09:10:25.285187: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
<br />[NbConvertApp] ERROR | Timeout waiting for execute reply (30s).



### Error 33, [Traceback at line 1126](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1126)<br />1126..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 405, in preprocess
<br />    nb, resources = super(ExecutePreprocessor, self).preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 69, in preprocess
<br />    nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 438, in preprocess_cell
<br />    reply, outputs = self.run_cell(cell, cell_index, store_history)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 571, in run_cell
<br />    if self._passed_deadline(deadline):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 541, in _passed_deadline
<br />    self._handle_timeout()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 504, in _handle_timeout
<br />    raise TimeoutError("Cell execution timed out")
<br />TimeoutError: Cell execution timed out



### Error 34, [Traceback at line 1179](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1179)<br />1179..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/html.py", line 95, in from_notebook_node
<br />    return super(HTMLExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/templateexporter.py", line 307, in from_notebook_node
<br />    nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 139, in from_notebook_node
<br />    nb_copy, resources = self._preprocess(nb_copy, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 316, in _preprocess
<br />    nbc, resc = preprocessor(nbc, resc)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
<br />    return self.preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 405, in preprocess
<br />    nb, resources = super(ExecutePreprocessor, self).preprocess(nb, resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/base.py", line 69, in preprocess
<br />    nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 438, in preprocess_cell
<br />    reply, outputs = self.run_cell(cell, cell_index, store_history)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 571, in run_cell
<br />    if self._passed_deadline(deadline):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 541, in _passed_deadline
<br />    self._handle_timeout()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/preprocessors/execute.py", line 504, in _handle_timeout
<br />    raise TimeoutError("Cell execution timed out")
<br />TimeoutError: Cell execution timed out



### Error 35, [Traceback at line 1230](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1230)<br />1230..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 14, in parse_json
<br />    nb_dict = json.loads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/__init__.py", line 354, in loads
<br />    return _default_decoder.decode(s)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 339, in decode
<br />    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 357, in raw_decode
<br />    raise JSONDecodeError("Expecting value", s, err.value) from None
<br />json.decoder.JSONDecodeError: Expecting value: line 2 column 1 (char 1)
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 36, [Traceback at line 1243](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1243)<br />1243..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 143, in read
<br />    return reads(buf, as_version, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 73, in reads
<br />    nb = reader.reads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 58, in reads
<br />    nb_dict = parse_json(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 17, in parse_json
<br />    raise NotJSONError(("Notebook does not appear to be JSON: %r" % s)[:77] + "...")
<br />nbformat.reader.NotJSONError: Notebook does not appear to be JSON: '\n# import library\nimport json\nimport...
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vision_mnist.py 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//vision_mnist.py to html



### Error 37, [Traceback at line 1280](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1280)<br />1280..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 14, in parse_json
<br />    nb_dict = json.loads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/__init__.py", line 354, in loads
<br />    return _default_decoder.decode(s)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 339, in decode
<br />    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 357, in raw_decode
<br />    raise JSONDecodeError("Expecting value", s, err.value) from None
<br />json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 38, [Traceback at line 1293](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1293)<br />1293..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 143, in read
<br />    return reads(buf, as_version, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 73, in reads
<br />    nb = reader.reads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 58, in reads
<br />    nb_dict = parse_json(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 17, in parse_json
<br />    raise NotJSONError(("Notebook does not appear to be JSON: %r" % s)[:77] + "...")
<br />nbformat.reader.NotJSONError: Notebook does not appear to be JSON: '# -*- coding: utf-8 -*-\n"""mnist mlmod...
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//benchmark_timeseries_m5.py 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//benchmark_timeseries_m5.py to html



### Error 39, [Traceback at line 1330](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1330)<br />1330..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 14, in parse_json
<br />    nb_dict = json.loads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/__init__.py", line 354, in loads
<br />    return _default_decoder.decode(s)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 342, in decode
<br />    raise JSONDecodeError("Extra data", s, end)
<br />json.decoder.JSONDecodeError: Extra data: line 1 column 3 (char 2)
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 40, [Traceback at line 1341](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1341)<br />1341..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 143, in read
<br />    return reads(buf, as_version, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 73, in reads
<br />    nb = reader.reads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 58, in reads
<br />    nb_dict = parse_json(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 17, in parse_json
<br />    raise NotJSONError(("Notebook does not appear to be JSON: %r" % s)[:77] + "...")
<br />nbformat.reader.NotJSONError: Notebook does not appear to be JSON: '"""\nM5 Forecasting Competition GluonTS...
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/benchmark_timeseries_m5.py 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example/benchmark_timeseries_m5.py to html



### Error 41, [Traceback at line 1378](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1378)<br />1378..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 14, in parse_json
<br />    nb_dict = json.loads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/__init__.py", line 354, in loads
<br />    return _default_decoder.decode(s)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 342, in decode
<br />    raise JSONDecodeError("Extra data", s, end)
<br />json.decoder.JSONDecodeError: Extra data: line 1 column 3 (char 2)
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 42, [Traceback at line 1389](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1389)<br />1389..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 143, in read
<br />    return reads(buf, as_version, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 73, in reads
<br />    nb = reader.reads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 58, in reads
<br />    nb_dict = parse_json(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 17, in parse_json
<br />    raise NotJSONError(("Notebook does not appear to be JSON: %r" % s)[:77] + "...")
<br />nbformat.reader.NotJSONError: Notebook does not appear to be JSON: '"""\nM5 Forecasting Competition GluonTS...
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/benchmark_timeseries_m4.py 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example/benchmark_timeseries_m4.py to html



### Error 43, [Traceback at line 1426](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1426)<br />1426..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 14, in parse_json
<br />    nb_dict = json.loads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/__init__.py", line 354, in loads
<br />    return _default_decoder.decode(s)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 339, in decode
<br />    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 357, in raw_decode
<br />    raise JSONDecodeError("Expecting value", s, err.value) from None
<br />json.decoder.JSONDecodeError: Expecting value: line 4 column 1 (char 3)
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 44, [Traceback at line 1439](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1439)<br />1439..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 143, in read
<br />    return reads(buf, as_version, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 73, in reads
<br />    nb = reader.reads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 58, in reads
<br />    nb_dict = parse_json(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 17, in parse_json
<br />    raise NotJSONError(("Notebook does not appear to be JSON: %r" % s)[:77] + "...")
<br />nbformat.reader.NotJSONError: Notebook does not appear to be JSON: '\n\n\ndef benchmark_m4() :\n    # This ...
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//arun_hyper.py 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//arun_hyper.py to html



### Error 45, [Traceback at line 1476](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1476)<br />1476..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 14, in parse_json
<br />    nb_dict = json.loads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/__init__.py", line 354, in loads
<br />    return _default_decoder.decode(s)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 339, in decode
<br />    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 357, in raw_decode
<br />    raise JSONDecodeError("Expecting value", s, err.value) from None
<br />json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 46, [Traceback at line 1489](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1489)<br />1489..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 143, in read
<br />    return reads(buf, as_version, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 73, in reads
<br />    nb = reader.reads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 58, in reads
<br />    nb_dict = parse_json(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 17, in parse_json
<br />    raise NotJSONError(("Notebook does not appear to be JSON: %r" % s)[:77] + "...")
<br />nbformat.reader.NotJSONError: Notebook does not appear to be JSON: '# import library\nimport json, copy\nfr...
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//benchmark_timeseries_m4.py 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//benchmark_timeseries_m4.py to html



### Error 47, [Traceback at line 1526](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1526)<br />1526..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 14, in parse_json
<br />    nb_dict = json.loads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/__init__.py", line 354, in loads
<br />    return _default_decoder.decode(s)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 339, in decode
<br />    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 357, in raw_decode
<br />    raise JSONDecodeError("Expecting value", s, err.value) from None
<br />json.decoder.JSONDecodeError: Expecting value: line 4 column 1 (char 3)
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 48, [Traceback at line 1539](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1539)<br />1539..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 143, in read
<br />    return reads(buf, as_version, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 73, in reads
<br />    nb = reader.reads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 58, in reads
<br />    nb_dict = parse_json(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 17, in parse_json
<br />    raise NotJSONError(("Notebook does not appear to be JSON: %r" % s)[:77] + "...")
<br />nbformat.reader.NotJSONError: Notebook does not appear to be JSON: '\n\n\ndef benchmark_m4() :\n    # This ...
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />jupyter nbconvert --execute https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_glass.py 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/09dbb573cf89ddf861ea945ff13f39f474d48070/mlmodels/example//lightgbm_glass.py to html



### Error 49, [Traceback at line 1576](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1576)<br />1576..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 14, in parse_json
<br />    nb_dict = json.loads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/__init__.py", line 354, in loads
<br />    return _default_decoder.decode(s)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 339, in decode
<br />    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/json/decoder.py", line 357, in raw_decode
<br />    raise JSONDecodeError("Expecting value", s, err.value) from None
<br />json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 50, [Traceback at line 1589](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1589)<br />1589..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/jupyter-nbconvert", line 8, in <module>
<br />    sys.exit(main())
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/jupyter_core/application.py", line 270, in launch_instance
<br />    return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/traitlets/config/application.py", line 664, in launch_instance
<br />    app.start()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 340, in start
<br />    self.convert_notebooks()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 510, in convert_notebooks
<br />    self.convert_single_notebook(notebook_filename)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 481, in convert_single_notebook
<br />    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/nbconvertapp.py", line 410, in export_single_notebook
<br />    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 179, in from_filename
<br />    return self.from_file(f, resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbconvert/exporters/exporter.py", line 197, in from_file
<br />    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 143, in read
<br />    return reads(buf, as_version, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/__init__.py", line 73, in reads
<br />    nb = reader.reads(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 58, in reads
<br />    nb_dict = parse_json(s, **kwargs)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/nbformat/reader.py", line 17, in parse_json
<br />    raise NotJSONError(("Notebook does not appear to be JSON: %r" % s)[:77] + "...")
<br />nbformat.reader.NotJSONError: Notebook does not appear to be JSON: '# import library\nimport os\nimport mlm...
