# Monitor Data Drift

## Overview

Changing trends in data over time can reduce the accuracy of the predictions made by a model. Monitoring for this *data drift* and retraining as necessary is an important way to ensure your machine learning solution continues to predict accurately.

## Open Jupyter

While you can use the **Notebooks** page in Azure Machine Learning studio to run notebooks, it's often more productive to use a more fully-featured notebook development environment like *Jupyter*.

> **Tip**: Jupyter Notebook is a commonly used open-source tool for data science. You can refer to the [documentation](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html) if you are unfamiliar with it.

1. In the LabVM browser open Azure Machine Learning Studio(https://ml.azure.com). If prompted, login using the credentials provided in the **Environment Details** tab. View the **Compute** page for your workspace.

    **Note**: If you get **Welcome to the studio** page when you login, Select **Subscription** and **Machine Learning workspace** available in the drop down then click on **Get Started**.

2. On the **Compute Instances** tab, start your compute instance if it is not already running.

3. When the compute instance is running, click the **Jupyter** link to open the Jupyter home page in a new browser tab. Be sure to open *Jupyter* and not *JupyterLab*.

    ![](images/jupyter.png)

## Monitor data drift for a dataset

In this exercise, the code to monitor data drift is provided in a notebook.

1. In the Jupyter home page, browse to the **Users/mslearn-dp100** or **Users/*{Username}*/mslearn-dp100** folder where you cloned the notebook repository.

2. Click on **Monitor Data Drift** (17 - Monitor Data Drift.ipynb) notebook to open in the new tab.

    ![](images/monitordata.png)

>**Note**: If you see a pop up saying **Kernel Not Found**, From the dropdown select the **Python 3.8 - AzureML** and **Set Kernel**

   ![](images/kernel-not-found.png)

3. Then read the notes in the notebook, running each code cell in turn. To run each cell select and click on **Run** on the menu.

>**Note**: While running the cells, you might get an error saying **Datastore.upload** is deprecated. Ignore it and carry on with running the next cells.

> **Note**: Kindly wait after you run each code cell until the output for the code is shown and CPU utilization is 0% else the succeeding code cells will not get executed.

4. When you have finished running the code in the notebook, on the **File** menu, click **Close and Halt** to close it and shut down its Python kernel. Then close all Jupyter browser tabs.
