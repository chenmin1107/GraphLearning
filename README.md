# GraphLearning
Explore the state-of-the-art graph learning algorithms

# Environment set up
1. Intall [Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/)
2. Create a new cond env. Add the following config to ~/.condarc
```
auto_activate_base: false
auto_update_conda: false
```
3. Install [DGL](https://www.dgl.ai/pages/start.html)
4. Install [pyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
5. Register [conda env in jupter notebook kernel](https://towardsdatascience.com/get-your-conda-environment-to-show-in-jupyter-notebooks-the-easy-way-17010b76e874)
```
conda install ipykernel
[Register] ipython kernel install --user --name=new-env
[Unregister] jupyter kernelspec uninstall new-env
```
