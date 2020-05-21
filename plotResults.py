import json
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import re

def main(args):

	# Load data
	with open(args.data, "rt") as f:
		data = json.load(f)

	# Store results separately
	results = data["results"]

	# Get x axis (epochs) values
	x = list(range(len(results["avg"])))

	# Plot cumulative average alone
	plt.figure()

	plt.plot(x, results["avg"])

	plt.title("Cumulative Average Reward")
	plt.xlabel("Time-Step")
	plt.ylabel("Avergage reward at time-step")

	plt.savefig(re.sub('\.json$', '_avg', args.data) +".png", dpi=900)

	#plt.show()

if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument(
		"--data",
		type = lambda x : x + ".json" if not x.endswith(".json") else x,
		default = "results.json",
		help = "Name of json file from which to get results"
	)

	args = parser.parse_args()

	main(args)