

all:
	nvcc assignment2.cu -o assignment2

clean:
	rm assignment2
	rm A_exlusive_scan_results.txt
	rm B_repeated_index_results.txt
	rm C_no_repeats_results.txt
