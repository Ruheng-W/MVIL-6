# MVIL-6: accurate identification of IL-6-induced peptides using multi-view feature learning

In this study, we developed a deep learning model called MVIL-6 for predicting IL-6-inducing peptides. Comparative results demonstrated the outstanding performance and robustness of MVIL-6. Specifically, we employ a pre-trained protein language model MG-BERT and the Transformer model to process two different sequence-based descriptors and integrate them with a fusion module to improve the prediction performance. The ablation experiment demonstrated the effectiveness of our fusion strategy for the two models. In addition, to provide good interpretability of our model, we explored and visualized the amino acids considered important for IL-6-induced peptide prediction by our model. Finally, a case study presented using MVIL-6 to predict IL-6-induced peptides in the SARS-CoV-2 spike protein shows that MVIL-6 achieves higher performance than existing methods and can be useful for identifying potential IL-6-induced peptides in viral proteins.

## Experimental settings
In our study, the whole deep learning models were trained globally by the Adam algorithm with a learning rate l = 1e-5 to minimize the cost function Loss. The training epoch is set to 100, and 50 for MGBERT and Transformer, and perform best in the around 85 and 27 epochs. All the experiments were performed based on Nvidia RTX 3090 GPUs and implemented by python based on PyTorch and Tensorflow.

## Contact

For further questions or details, reach out to Ruheng Wang (wangruheng@mail.sdu.edu.cn)
