The script SK_PLL.py incorporates Kevin Shen's kmag prunning and Nick Sherck's PLL implementation for calculating the structure factor for up to 2 species. Initial implementation written by K. Delaney @ MRL UCSB.

ToRun:
python SK_PLL.py -t CGTraj.dcd -p CGTraj.pdb -m 1 --pruneRes 0.05 --pruneNum 50 -d SK_PLL_ExampleOutput -s 1 -w 0 -np 10

Additional Folders:
BaselineSk - Contains baseline Sk data from running old SK.py script before PLL added.
ExampleTraj - Contains reference trajectories that correspond to the Sk files in BaselineSK and SK_PLL_ExampleOutput.
SK_PLL_Example - Contains all outputs from running the command above on the ExampleTraj using SK_PLL.py
