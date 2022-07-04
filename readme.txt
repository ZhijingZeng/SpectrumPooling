This is to redo the experiment in the paper https://ieeexplore.ieee.org/document/9677426.
Here is a summary of what we learnt in it as a project report in the course of Game theory.
https://www.overleaf.com/read/bmgjkyhkfsnr

To run the program, please download jupytext using 
$ pip install jupytext --upgrade
or
$ conda install jupytext -c conda-forge
($ is to indicate it is a command, please do not type it)
Double click the .py file in jupyterLab or jupyter notebook and then it will generate a .ipynb file automatically. The use of jupytext is for better version control and collabration
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
setting up the environment in Lab M338
1.download and install anaconda:

from https://www.anaconda.com/ download linux version anaconda

in terminal: $ bash Anaconda3-2022.05-Linux-x86_64.sh

enter yes to every question and proceed to the end.

2.test anaconda:

open vscode (already installed)

press ctrl+` (open terminal)

change terminal to "bash" by clicking the dropdown button on the upper right side of the poped up window.

type $ anaconda-navigator the graphic menu should appear!

3.configure git

First, to use Git on the command line, you will need to download, install, and configure Git on your computer.

(git is already installed in the Lab, but I did not successfully set it up this way)

You can also install GitHub CLI to use GitHub from the command line.

https://docs.github.com/en/get-started/quickstart/set-up-git

$ conda install gh --channel conda-forge

After downloaded, type

$gh auth login

follow the instruction and then it is ready to use.

4.In my vscode, jupyter notebook can not be used. I walked around it by

type $anaconda-navigator in the terminal (the same terminal as in 2.test anaconda)

when the app opened, click "Launch" below vscode

in the new window of vscode, it can read jupyter-notebook.