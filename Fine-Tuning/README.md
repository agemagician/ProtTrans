Important Notes:
----------------

1. **For easy setup of a conda environment to run the notebooks you can use the finetuning.yml File provided in this folder**

    check here for [setting up env from a yml File](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

2. **For details, check information inside the notebooks**

3. **Additional advice**
    - if GPU memory overflow occurs, reduce batch size to 1 and simulated the desired batch size using gradient accumulation
    - if GPU memory still overflows, use mixed percision training and/or remove(or truncate) long sequences 
    - multi GPU training is not supported by deepspeed running from a notebook. Run it in a script using [deepspeed](https://huggingface.co/docs/transformers/main_classes/deepspeed#deployment-with-multiple-gpus) or a similar distributed launcher.
    
