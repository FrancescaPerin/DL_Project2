import json
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np

def process_vals(args):
	# Load data

	data_all = []

	for file in args.data:
		with open(file, "rt") as f:
			data_all.append(json.load(f))

	# Store results separately
	results_all = [data["results"] for data in data_all]

	# Get x axis (epochs) values
	x = list(range(len(results_all[0]["avg"])))

	# Average results
	avg_all = [results["avg"] for results in results_all]

	avg_all = np.vstack(avg_all)

	avg = avg_all.mean(0)
	avg_std = avg_all.std(0)

	return x, avg, avg_std

def plot_line(x, avg, avg_std = None):

	plt.plot(x, avg)

	if avg_std is not None:
		plt.fill_between(x, avg+avg_std, avg-avg_std, alpha = 0.2)

def main(args):

	x, avg, avg_std = process_vals(args)

	# Plot cumulative average alone
	plt.figure()

	plot_line(x, avg, avg_std if args.std else None)
	
	plt.title("Average Reward across 4 runs")
	plt.xlabel("Time-Step")
	plt.ylabel("Avergage reward at time-step")

	if args.save_to is not None:
		plt.savefig(args.save_to, dpi=900)

	if args.show:
		plt.show()

if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument(
		"--data",
		type = lambda x : x + ".json" if not x.endswith(".json") else x,
		nargs = "+",
		default = ["results.json"],
		help = "Name of json file from which to get results"
	)

	parser.add_argument(
		"--save_to",
		type = lambda x : x + ".png" if not x.endswith(".png") else x,
		default = None,
		help = "If given, plot is saved in a file with this name"
	)

	parser.add_argument(
		"--show",
		type = lambda x : x.lower() == "true",
		default = False,
		help = "If true shows the plot"
	)

	parser.add_argument(
		"--std",
		type = lambda x : x.lower() == "true",
		default = False,
		help = "If true shows the std on the plot"
	)

	args = parser.parse_args()

	main(args)