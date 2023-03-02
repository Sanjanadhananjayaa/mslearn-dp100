

# Clone the repo to perform the lab

1. **It is necessary that after completing this exercise, then proceed to the next one**.

1. After signing into the azure portal using the credentials provided, Search for **Resource Groups** in the search bar.

1. Then you will find the **dp-100-<inject key="DeploymentID" enableCopy="false"/>**. Click on **dp-100-<inject key="DeploymentID" enableCopy="false"/>** to find the resources for the lab.

    ![](images/img1.png)

1. Find the machine learning workspace named **quick-start-ws-<inject key="DeploymentID" enableCopy="false"/>**, Click to open it.

    ![](images/img2.png)
    
1. Click on **Launch Studio** to open the machine learning studio in a new tab. If any pop-up appears close them.

    ![](images/img3.png)
    
1. Find **compute** from the Left navigation pane.

    ![](images/img4.png)
    
1. Under **compute instances** you can find **notebook<inject key="DeploymentID" enableCopy="false"/>**. Click on **Jupyter** (make sure to click on **Jupyter** not **JupyterLab**) to run the notebook to perform the lab. A lot of data science and machine learning experimentation is performed by running code in *notebooks*. Your compute instance includes fully featured Python notebook environments (*Jupyter* and *JuypyterLab*) that you can use for extensive work.

    ![](images/img5.png)
    
1. A pop-up appears, select **Yes, I Understand** and click on **Continue**.

    ![](images/img6.png)
    
1. The Jupyter will open in a new window, Click on **New** then select **Terminal** to open a new terminal.

    ![](images/img7.png)
    
    ![](images/img8.png)

1. Terminal will open in a new window, Enter the following commands to clone a Git repository containing notebooks, data, and other files to your workspace:

    ```bash
    cd Users
    git clone https://github.com/MicrosoftLearning/mslearn-dp100
    ```

1. After running the commands you will get the output as shown below image. Then close the terminal window browser tab and go back to the jupyter window.

    ![](images/img9.png)

1. Click **&#8635;** to refresh the view and verify that a new **Users/mslearn-dp100** folder has been created. This folder contains multiple **.ipynb** notebook files.

   > **Tip**: To run a code cell in the notebook, select the cell you want to run and then use the **&#9655;** button to run it.

1. If asked to provide **compute cluster** while running the notebook, then please provide compute cluster as **aml-compute**.

1. Click on **Next** to continue to the next module in this lab.
