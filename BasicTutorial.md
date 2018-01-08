# Train a Basic Agent

**Step 1:** First, from the command line, clone the main OpenMined Repo, PySyft, and this repository.

> git clone https://github.com/OpenMined/OpenMined
> git clone https://github.com/OpenMined/PySyft.git
> git clone https://github.com/OpenMined/ml-agents.git

**Step 2:** Then, from the command line, build PySyft.

> cd PySyft
> python setup.py clean install
> cd ../

**Step 3:** Next, we need to setup OpenMined/OpenMined. So, walk through the [Readme](https://github.com/OpenMined/OpenMined)

**Step 4:** If you haven't already, turn on Metal and Background process.
- Start Unity Application (download and install free ver if you don’t have it: https://unity3d.com/get-unity/download )
- Within Unity editor select: Edit -> Project Settings -> Player
- In the Inspector, scroll down until you find “Metal Editor Support” and click the checkbox to turn it on. (skip this if not on Mac OSX)
- In the Inspector, expand “Resolution” and check the box for “Run in Background”

**Step 5:* Select `Basic` Environment
- In the `Project` panel, single click /Assets/ML-Agents/Examples/Basic
- With the previous folder selected, double slick the `Scene` object.

**Step 6:** Wire up SyftServer to Academy/Brain (if not already)
- In the `Hierarchy` panel, open the `Academy` dropdown and single-click `Brain`
- In the `Inspector` look for whether or not Syft Server is attached as a script.
- If Syft Server is not a script in the Inspector when `Brain` is selected:
  - in the `Project` panel select /Assets/OpenMined/Network/Servers/SyftServer
  - Drag it to the Inspector to attach it as a script (`Brain` should still be selected in the `Hierarchy)
- In the SyftServer script now attached in the `Inspector`, see if the `Shader` field contains "FloatTensorShaders". If not:
  - in the `Project` panel select /Assets/OpenMined/Syft/Tensor/Ops/Shaders/FloatTensorShaders
  - Drag it to the Inspector and drop it in the `Shader` field.
  
**Step 7:** Start the Environment
- Push the play button. You should start to see a blue box racing to the left repeatedly (this means no neural network is attached)

**Step 8:** Start Jupyter Notebook
- In the OpenMined/OpenMined repository (the first thing you cloned in step 1), navigate to the root and run
> jupyter notebook

**Step 9:** Open the Demo Notebook
- In the notebooks/ml-agents folder there is a notebook [Basic Ml Agent.ipynb](https://github.com/OpenMined/OpenMined/blob/master/notebooks/ml-agents/Basic%20ML%20Agent.ipynb)
- Run all cells in the notebook

**Step 10** Observe Learning
- The blue box should now be moving randomly left and right. If it learns, it should end up picking only one direction again after a few seconds. Usually it will pick right. Sometimes it will pick left (we'll fix this later).
