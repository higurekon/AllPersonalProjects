#parser.add_option("-r","--radius",type="int",help="enter a number for the starting radius for the neighboring neurons.",dest="radius")
#parser.add_option("-n","--number",type="int",help="enter the number of neurons desired for self organizing map.",dest="number")
#parser.add_option("-e","--epoch",type="int",help="enter the number of iterations desired for self organizing map.",dest="epoch")
#parser.add_option("-l","--learn-rate",help="enter a number for the learning rate of the self organizing map.",dest="learn")
#parser.add_option("-i","--input-vector",help="enter an input vector of data. required.",dest="vectors")
#parser.add_option("-o","--output-vector",help="enter an output file name. will output to a file called 'som_bins.txt' by default in the current path.",dest="out")
#parser.add_option("-t","--test",action="store_true",help="if given, program will only print the first 4 vectors in the input so you can see your data as the program is about to read it.",dest="test")
#parser.add_option("-c","--cluster",type="int",help="enter a radius for the number of neurons to include that are closest to your bestmatch for each vector",dest="cluster")

time python scripts/som_torus_cluster.py -i esom_outs/output1000.txt -o outputs/som_bins_output1000_cluster_r8_e150.txt -c 8 -e 150 &

time python scripts/som_torus.py -i esom_outs/output1000.txt -o outputs/som_bins_output1000_e150.txt -e 150 &

time python scripts/som.py -i esom_outs/output1000.txt -o outputs/som_bins_output1000_notorus_e150.txt -e 150 &

time python scripts/som_torus_cluster.py -i esom_outs/output1000.txt -o outputs/som_bins_output1000_cluster_r15_e150.txt -c 15 -e 150 &

time python scripts/som_torus_cluster.py -i esom_outs/output1500.txt -o outputs/som_bins_output1500_cluster_r15_e150.txt -c 15 -e 150 &

time python scripts/som_torus_cluster.py -i esom_outs/output1500.txt -o outputs/som_bins_output1500_cluster_r15_e150_l0.1.txt -c 15 -e 150 -l 0.1 &

python scripts/process_cluster.py outputs/som_bins_output1500_cluster_r15_e150_l0.1.txt outputs/som_bins_output1500_cluster_r15_e150.txt outputs/som_bins_output1000_cluster_r15_e150.txt outputs/som_bins_output1000_cluster_r8_e150.txt

time python scripts/som_torus_cluster.py -i esom_outs/output1500.txt -o outputs/som_bins_output1500_cluster_c10_e100.txt -c 50 -e 200 & > outputs/runtime_som_bins_output1500_cluster_c10_e100.txt

time python scripts/som_torus_cluster.py -i esom_outs/output1500.txt -o outputs/som_bins_output1500_cluster_c1_e100.txt -c 1 -e 200 & > outputs/runtime_som_bins_output1500_cluster_c1_e100.txt

wait

python scripts/process_cluster.py outputs/som_bins_output1500_cluster_c10_e100.txt

python scripts/process_cluster.py outputs/som_bins_output1500_cluster_c1_e100.txt

python scripts/process_cluster.py plots/kmeans_bins_thermatoga.txt plots/kohonen_bins_thermatoga.txt

python scripts/process_cluster.py outputs/som_bins_output1500_cluster_c50_e200.txt outputs/som_bins_output1500_cluster_c1_e200.txt
