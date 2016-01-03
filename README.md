README

Requires installation of Anaconda distribution

##########Coevolutioner

Example usage of coevolutioner.py is:
python coevolutioner.py

Check coevolutioner_workspace for the outputs.
Generates "frame_n.png" plots that depict the movement of the organisms on the field and the plants that currently populate it.
brains.png and diets.png show the dominating traits that fall into these categories.
org_v_plt.png shows the prevalence of organisms and plants over time.
species_progression_plot.png shows the prevalence of each unique species over time.
species_stats_plot.png shows the difference between the number of species and the average members per species over time.
traits_plot.png shows the average value of all traits over time.

##########Self-Organizing Map

Example usage of som_torus_cluster.py is:
python som_torus_cluster.py -i som_workspace/output1000.txt -o som_workspace/som_bins_output1000_cluster_r8_e150.txt -c 8 -e 150 &
