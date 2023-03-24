# Use Automated Machine Learning from the SDK

## Overview

Determining the right algorithm and preprocessing transformations for model training can involve a lot of guesswork and experimentation.

In this exercise, you'll use automated machine learning to determine the optimal algorithm and preprocessing steps for a model by performing multiple training runs in parallel.

## Open Jupyter

While you can use the **Notebooks** page in Azure Machine Learning studio to run notebooks, it's often more productive to use a more fully-featured notebook development environment like *Jupyter*.

> **Tip**: Jupyter Notebook is a commonly used open-source tool for data science. You can refer to the [documentation](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html) if you are unfamiliar with it.

1. In the LabVM browser open Azure Machine Learning Studio(https://ml.azure.com). If prompted, login using the credentials provided in the **Environment Details** tab. View the **Compute** page for your workspace.

    **Note**: If you get **Welcome to the studio** page when you login, Select **Subscription** and **Machine Learning workspace** available in the drop down then click on **Get Started**.

2. On the **Compute Instances** tab, start your compute instance if it is not already running.

3. When the compute instance is running, click the **Jupyter** link to open the Jupyter home page in a new browser tab. Be sure to open *Jupyter* and not *JupyterLab*.

    ![](images/jupyter.png)

## Use the SDK to run an automated machine learning experiment

In this exercise, the code to run an automated machine learning experiment is provided in a notebook.

1. In the Jupyter home page, browse to the **Users/mslearn-dp100** or **Users/*{Username}*/mslearn-dp100** folder where you cloned the notebook repository.

2. Click on **Use Automated Machine Learning** (12 - Use Automated Machine Learning.ipynb) notebook to open in the new tab.

    ![](images/runml.png)

3. Then read the notes in the notebook, running each code cell in turn. To run each cell select and click on **Run** on the menu and follow the below instruction mentioned in the image to identify whether run has completed or still running the code.

    ![](images/Note.png)
    
4. When you have finished running the code in the notebook, on the **File** menu, click **Close and Halt** to close it and shut down its Python kernel. Then close all Jupyter browser tabs.



