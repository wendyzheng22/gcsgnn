The end goal is to adapt each baseline into a single callable method.

The original_code is just for reference, and the code_to_clean folder contains the code that needs to be cleaned up and tested. In the end, both folders will be removed, so do not reference them, and move the code into the respective baseline python file (located in the baselines subdirectory).

main.py has some driver code to run the baseline, and the command to run the baseline is
```bash
python main.py -et baseline -bt [baseline_type]
```

If needed, check out cf_gnnexplainer for an example of what to do.
