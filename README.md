# Object-Contour-Detection-CNN
Object Contour Detection / Edge Detection CNN using Tensorflow 2 using the U-Net and UNet++ architectures.

* UNet Paper: https://arxiv.org/abs/1505.04597
* UNet++ Paper: https://arxiv.org/abs/1807.10165  

To run the program execute the following command in the command line:
```python main.py --[Command] --[Architecture]```

The allowed values for **[Command]** and **[Architecture]** are shown in the table below:

<table>
    <tr>
      <th>Command</th>
      <th>Architecture</th>
    </tr>
    <tr>
      <td>Help</td>
      <td>UNet</td>
    </tr>
    <tr>
      <td>Train</td>
      <td>UNet++</td>
    </tr>
     <tr>
      <td>Summary</td>
      <td>AutoEncoder1</td>
    </tr>
     <tr>
      <td>Evaluate</td>
      <td>AutoEncoder2</td>
    </tr>
     <tr>
      <td>Predict</td>
      <td></td>
    </tr>
</table>

**Command** refers to the functionality of the **Architecture** you which the utilise.

* **Train** trains the model on the training set.
* **Summary** outputs a summary of the architecture of the model.
* **Evaluate** gives the accuracy of the model on the test set.
* **Predict** generates images of the edges as predicted by the model.
